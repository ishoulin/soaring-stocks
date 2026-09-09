import json
import os
import re
import smtplib
from email.header import Header
from email.mime.text import MIMEText
import pandas as pd
import requests
import yfinance as yf


# ====================================================
# 1. ZenMomentum 算力引擎
# ====================================================
class ZenMomentumEngine:

  def __init__(self, state_file="arena_state.json"):
    self.state_file = state_file
    self.state = self._load_state()

  def _load_state(self):
    try:
      with open(self.state_file, "r") as f:
        return json.load(f)
    except FileNotFoundError:
      return {"slot_a": None}

  def _save_state(self):
    with open(self.state_file, "w") as f:
      json.dump(self.state, f, ensure_ascii=False, indent=2)

  def analyze_stock(self, df):
    window, lookback = 20, 60
    ma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()
    bb_width = (4 * std) / ma

    min_w = bb_width.rolling(window=lookback, min_periods=10).min()
    max_w = bb_width.rolling(window=lookback, min_periods=10).max()
    denom = (max_w - min_w).replace(0, 0.0001)

    df["compression_score"] = (
        (100 * (1 - (bb_width - min_w) / denom)).clip(0, 100).fillna(50)
    )

    price_range = ((df["High"] - df["Low"]) / df["Close"]).replace(0, 0.001)
    inst_buy_5d = df["Inst_Net_Buy"].rolling(window=5).sum()
    range_avg_5d = price_range.rolling(window=5).mean()
    raw_density = inst_buy_5d / range_avg_5d

    df["density_score"] = (
        raw_density.rolling(window=lookback, min_periods=10)
        .apply(
            lambda x: (
                pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
            )
        )
        .fillna(50)
    )

    df["energy_score"] = (df["compression_score"] * 0.6) + (
        df["density_score"] * 0.4
    )
    return df

  def run_daily_arena(self, stock_data_dict):
    today_slot_a, today_slot_b = None, None
    all_scores = []

    prev_a = self.state.get("slot_a")
    if prev_a:
      symbol = prev_a["symbol"]
      df = stock_data_dict.get(symbol)
      if df is not None and not df.empty:
        latest_close = df.iloc[-1]["Close"]
        if latest_close >= prev_a["stop_loss"]:
          prev_a["streak"] += 1
          prev_a["stop_loss"] = max(
              prev_a["stop_loss"], df.iloc[-5:]["Low"].min()
          )
          today_slot_a = prev_a

    for symbol, df in stock_data_dict.items():
      if prev_a and symbol == prev_a["symbol"]:
        continue
      analyzed_df = self.analyze_stock(df)
      latest = analyzed_df.iloc[-1]
      score = round(latest["energy_score"], 1)

      if not pd.isna(score):
        all_scores.append({
            "symbol": symbol,
            "score": score,
            "close": round(latest["Close"], 2),
            "stop_loss": round(df.iloc[-5:]["Low"].min(), 2),
        })

    all_scores.sort(key=lambda x: x["score"], reverse=True)

    if all_scores and all_scores[0]["score"] >= 90.0:
      best = all_scores[0]
      today_slot_b = {
          "symbol": best["symbol"],
          "score": best["score"],
          "stop_loss": best["stop_loss"],
      }

    if not today_slot_a and self.state.get("pending_slot_b"):
      pending = self.state["pending_slot_b"]
      today_slot_a = {
          "symbol": pending["symbol"],
          "streak": 1,
          "stop_loss": pending["stop_loss"],
      }

    self.state["slot_a"] = today_slot_a
    self.state["pending_slot_b"] = today_slot_b
    self._save_state()

    return today_slot_a, today_slot_b, all_scores[:5]


def fetch_tw_stock_tickers():
  url_listed = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
  url_otc = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
  stock_dict = {}
  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
      )
  }

  for url, suffix in [(url_listed, ".TW"), (url_otc, ".TWO")]:
    try:
      res = requests.get(url, headers=headers)
      res.encoding = "cp950"
      df_list = pd.read_html(res.text)
      df = df_list[0]
      df.columns = df.iloc[0]
      for _, row in df[1:].iterrows():
        raw_name = str(row.get("有價證券代號及名稱", ""))
        match = re.match(r"^(\d{4})\s+(.+)$", raw_name.strip())
        if match and not match.group(1).startswith("00"):
          stock_dict[f"{match.group(1)}{suffix}"] = match.group(2)
    except Exception:
      pass
  return stock_dict


# ====================================================
# 2. Email 自動發送函式
# ====================================================
def send_email_report(report_text):
  sender = os.getenv("EMAIL_USER", "").strip()
  password = os.getenv("EMAIL_PASS", "").strip()
  receiver = os.getenv("EMAIL_RECEIVER", "").strip() or sender
  
  if not sender or not password:
    print("⚠️ 未設定 Email 環境變數，跳過發信步驟。")
    print(report_text)
    return

  msg = MIMEText(report_text, "plain", "utf-8")
  msg["Subject"] = Header("⚡ ZenMomentum 每日台股雙槽戰報", "utf-8")
  msg["From"] = sender
  msg["To"] = receiver

  try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
      server.login(sender, password)
      server.sendmail(sender, [receiver], msg.as_string())
    print("📧 Email 戰報已成功發送！")
  except Exception as e:
    print(f"❌ 發信失敗: {e}")


# ====================================================
# 3. 主程式執行（已加入：剔除 2 年高點回落 > 20% 標的）
# ====================================================
if __name__ == "__main__":
  tw_stocks = fetch_tw_stock_tickers()
  symbol_list = list(tw_stocks.keys())
  all_stock_data = {}
  batch_size = 50

  total_fetched_count = len(symbol_list)
  print(f"🔍 成功抓取全台股清單：共 {total_fetched_count} 檔標的")

  for i in range(0, total_fetched_count, batch_size):
    chunk = symbol_list[i : i + batch_size]
    try:
      # 1. 時間範圍由 6mo 改為 2y，確保抓得到 2 年內的最高價
      data = yf.download(
          chunk, period="2y", group_by="ticker", threads=False, progress=False
      )
      for symbol in chunk:
        try:
          df = (
              data[symbol].copy()
              if isinstance(data.columns, pd.MultiIndex)
              else data.copy()
          )
          df = df.dropna(subset=["Close"])

          # 需有至少半年的 K 線資料，且近 5 日均量 > 500 張
          if not df.empty and len(df) >= 120:
            if df["Volume"].tail(5).mean() > 500000:

              # 2. 計算 2 年高點回落幅度
              high_2y = df["High"].max()  # 過去 2 年最高價
              current_close = df["Close"].iloc[-1]  # 當前收盤價
              drawdown_2y = (high_2y - current_close) / high_2y  # 回落比例

              # 核心過濾：只保留回落 <= 20% (相當於股價維持在 2 年最高點的 80% 以上)
              if drawdown_2y <= 0.20:
                df["Inst_Net_Buy"] = df["Volume"] * 0.2
                all_stock_data[symbol.split(".")[0]] = df

        except Exception:
          continue
    except Exception:
      pass

  valid_scanned_count = len(all_stock_data)
  print(
      f"✅ 完成數據清洗：共 {valid_scanned_count} 檔標的符合條件（均量 >"
      " 500張 且 距離2年高點回落 < 20%）"
  )

  # 後續的 engine 運算與戰報發送維持不變...
  
  engine = ZenMomentumEngine()
  slot_a, slot_b, top_5 = engine.run_daily_arena(all_stock_data)

  # 組裝 Email 戰報內文（加上資料筆數統計）
  report = (
      f"📈 【ZenMomentum 盤後數據掃描】\n"
      f"• 全台股掃描總數：{total_fetched_count} 檔\n"
      f"• 符合流動性標的：{valid_scanned_count} 檔（近5日均量>500張）\n"
      f"{'='*35}\n\n"
      f"📊 【盤後蓄能 Top 5】\n"
      f"{'='*35}\n"
  )

  for rank, cand in enumerate(top_5, 1):
    s = cand["symbol"]
    name = tw_stocks.get(f"{s}.TW", tw_stocks.get(f"{s}.TWO", ""))
    report += (
        f"第 {rank} 名 | {s} {name} | 能量: {cand['score']}% | 收盤: ${cand['close']}\n"
    )

  report += "\n🏆 【每日雙槽戰報】\n" + "=" * 35 + "\n"
  if slot_a:
    s = slot_a["symbol"]
    name = tw_stocks.get(f"{s}.TW", tw_stocks.get(f"{s}.TWO", ""))
    report += (
        f"👑 [Slot A 衛冕者] {s} {name}\n   連霸: {slot_a['streak']} 天 |"
        f" 防守點: ${slot_a['stop_loss']:.2f}\n"
    )
  else:
    report += "👑 [Slot A 衛冕者] 目前空缺\n"

  if slot_b:
    s = slot_b["symbol"]
    name = tw_stocks.get(f"{s}.TW", tw_stocks.get(f"{s}.TWO", ""))
    report += (
        f"⚡ [Slot B 挑戰者] {s} {name}\n   能量: {slot_b['score']}% |"
        f" 建議防守: ${slot_b['stop_loss']:.2f}\n"
    )
  else:
    report += "⚡ [Slot B 挑戰者] 無標的跨越 90 分發動線\n"

  send_email_report(report)
