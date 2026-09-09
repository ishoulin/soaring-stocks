import datetime
import os
import smtplib
from email.header import Header
from email.mime.text import MIMEText
import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ====================================================
# 1. 核心評分引擎 (ZenMomentum Engine)
# ====================================================
class ZenMomentumEngine:

  def calculate_score(self, df):
    """計算強勢打底與能量評分"""
    close = df['Close']
    volume = df['Volume']

    # 1. 動能分數 (近 20 日相對高檔)
    low_20 = df['Low'].tail(20).min()
    high_20 = df['High'].tail(20).max()
    pos_score = (
        (close.iloc[-1] - low_20) / (high_20 - low_20 + 1e-6)
    ) * 40  # 占 40 分

    # 2. 量能集中度 (近 5 日均量 vs 近 60 日均量)
    v_5 = volume.tail(5).mean()
    v_60 = volume.tail(60).mean()
    vol_ratio = min(v_5 / (v_60 + 1e-6), 2.0)
    vol_score = (vol_ratio / 2.0) * 30  # 占 30 分

    # 3. 價格穩定度 (近 20 日波動振幅，越低越好)
    range_20 = (high_20 - low_20) / (low_20 + 1e-6)
    stability_score = max(0, (1 - range_20 / 0.20)) * 30  # 占 30 分

    total_score = round(pos_score + vol_score + stability_score, 1)
    return min(total_score, 99.9)

  def run_daily_arena(self, stock_dict):
    candidates = []
    for symbol, df in stock_dict.items():
      try:
        score = self.calculate_score(df)
        close = round(float(df['Close'].iloc[-1]), 2)
        # 以近 20 日最低價做為基礎防守點
        stop_loss = round(float(df['Low'].tail(20).min() * 0.98), 2)

        candidates.append({
            'symbol': symbol,
            'score': score,
            'close': close,
            'stop_loss': stop_loss,
        })
      except Exception:
        continue

    # 依能量分數排序
    candidates.sort(key=lambda x: x['score'], reverse=True)
    top_5 = candidates[:5]

    slot_a = top_5[0] if len(top_5) > 0 else None
    if slot_a:
      slot_a['streak'] = 1

    slot_b = (
        top_5[1] if len(top_5) > 1 and top_5[1]['score'] >= 90.0 else None
    )

    return slot_a, slot_b, top_5


# ====================================================
# 2. 抓取全台股清單 (加入 timeout 防卡死)
# ====================================================
def fetch_tw_stock_tickers():
  stocks = {}
  headers = {
      'User-Agent': (
          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
      )
  }
  urls = [
      ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2', '.TW'),  # 上市
      ('https://isin.twse.com.tw/isin/C_public.jsp?strMode=4', '.TWO'),  # 上櫃
  ]

  for url, suffix in urls:
    try:
      res = requests.get(url, headers=headers, timeout=10)
      res.encoding = 'big5'
      df_list = pd.read_html(res.text)
      if df_list:
        df = df_list[0]
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        for entry in df['有價證券代號及名稱'].dropna():
          parts = entry.split('\u3000')
          if len(parts) == 2 and len(parts[0]) == 4 and parts[0].isdigit():
            stocks[f'{parts[0]}{suffix}'] = parts[1]
    except Exception as e:
      print(f'⚠️ 抓取 {suffix} 清單失敗: {e}')

  return stocks


# ====================================================
# 3. Email 自動發送函式
# ====================================================
def send_email_report(report_text):
  sender = os.getenv('EMAIL_USER', '').strip()
  password = os.getenv('EMAIL_PASS', '').strip()
  receiver = os.getenv('EMAIL_RECEIVER', '').strip() or sender

  if not sender or not password:
    print('⚠️ 未設定 Email 環境變數，跳過發信步驟。')
    print(report_text)
    return

  msg = MIMEText(report_text, 'plain', 'utf-8')
  msg['Subject'] = Header('⚡ ZenMomentum 每日台股強勢築底戰報', 'utf-8')
  msg['From'] = sender
  msg['To'] = receiver

  try:
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
      server.login(sender, password)
      server.sendmail(sender, [receiver], msg.as_string())
    print('📧 Email 戰報已成功發送！')
  except Exception as e:
    print(f'❌ 發信失敗: {e}')


# ====================================================
# 4. 主程式執行 (包含流動性過濾 + 突破 5 年新高檢測)
# ====================================================
if __name__ == '__main__':
  tw_stocks = fetch_tw_stock_tickers()
  symbol_list = list(tw_stocks.keys())
  total_fetched_count = len(symbol_list)
  print(f'🔍 成功抓取全台股清單：共 {total_fetched_count} 檔標的')

  batch_size = 50
  stage1_passed_symbols = []

  # --- 第一階段：快速篩選成交量 (6 個月數據) ---
  print('🚀 [第一階段] 快速掃描流動性（均量 > 500張）...')
  for i in range(0, total_fetched_count, batch_size):
    chunk = symbol_list[i : i + batch_size]
    try:
      data = yf.download(
          chunk, period='6mo', group_by='ticker', threads=False, progress=False
      )
      for symbol in chunk:
        try:
          df = (
              data[symbol].copy()
              if isinstance(data.columns, pd.MultiIndex)
              else data.copy()
          )
          df = df.dropna(subset=['Close'])
          if not df.empty and len(df) >= 30:
            if df['Volume'].tail(5).mean() > 500000:
              stage1_passed_symbols.append(symbol)
        except Exception:
          continue
    except Exception:
      pass

  print(f'✅ 第一階段完成：共 {len(stage1_passed_symbols)} 檔標的符合流動性條件')

  # --- 第二階段：精準下載 5 年數據 + 檢測「2年高點回落 <= 20%」與「突破5年新高」 ---
  print(
      '🎯 [第二階段] 精準檢測「2年高點回落 <= 20%」與「突破 5 年新高」標記...'
  )
  all_stock_data = {}
  is_5y_high_map = {}  # 紀錄是否突破 5 年新高

  for i in range(0, len(stage1_passed_symbols), batch_size):
    chunk = stage1_passed_symbols[i : i + batch_size]
    try:
      data_5y = yf.download(
          chunk, period='5y', group_by='ticker', threads=False, progress=False
      )
      for symbol in chunk:
        try:
          df = (
              data_5y[symbol].copy()
              if isinstance(data_5y.columns, pd.MultiIndex)
              else data_5y.copy()
          )
          df = df.dropna(subset=['Close'])

          if not df.empty and len(df) >= 120:
            # 取近 2 年 (約 500 個交易日) 最高價計算回落
            high_2y = df['High'].tail(500).max()
            current_close = df['Close'].iloc[-1]
            drawdown_2y = (high_2y - current_close) / high_2y

            # 核心條件：2 年高點回落 <= 20%
            if drawdown_2y <= 0.20:
              # 計算 5 年最高價 (不含今日，避免自己跟自己比)
              high_5y_prev = df['High'].iloc[:-1].max()

              # 判斷今日收盤價是否「突破 5 年最高價」
              is_breakout = current_close >= high_5y_prev

              clean_symbol = symbol.split('.')[0]
              all_stock_data[clean_symbol] = df
              is_5y_high_map[clean_symbol] = is_breakout
        except Exception:
          continue
    except Exception:
      pass

  valid_scanned_count = len(all_stock_data)
  print(
      f'🎉 篩選完成：最終共 {valid_scanned_count} 檔標的符合「高檔強勢築底」型態！'
  )

# --- 執行引擎計算與產生戰報 ---
    engine = ZenMomentumEngine()
    slot_a, slot_b, top_5 = engine.run_daily_arena(all_stock_data)

    report = (
        f'📈 【ZenMomentum 盤後數據掃描】\n'
        f'• 全台股掃描總數：{total_fetched_count} 檔\n'
        f'• 強勢築底合格標的：{valid_scanned_count} 檔（回落<20% + 均量>500張）\n'
        f"{'='*35}\n\n"
        f'📊 【盤後強勢 Top 5】\n'
        f"{'='*35}\n"
    )

    for rank, cand in enumerate(top_5, 1):
      s = cand['symbol']
      name = tw_stocks.get(f'{s}.TW', tw_stocks.get(f'{s}.TWO', ''))
      # 改為明確顯示 Yes 或 No
      is_high = is_5y_high_map.get(s, False)
      tag = ' | 突破5年新高: Yes' if is_high else ' | 突破5年新高: No'
      report += (
          f'第 {rank} 名 | {s} {name} | 能量: {cand["score"]}% | 收盤:'
          f' ${cand["close"]}{tag}\n'
      )

    report += '\n🏆 【每日雙槽戰報】\n' + '=' * 35 + '\n'
    if slot_a:
      s = slot_a['symbol']
      name = tw_stocks.get(f'{s}.TW', tw_stocks.get(f'{s}.TWO', ''))
      is_high = is_5y_high_map.get(s, False)
      tag = ' | 突破5年新高: Yes' if is_high else ' | 突破5年新高: No'
      report += (
          f'👑 [Slot A 衛冕者] {s} {name}{tag}\n   連霸: {slot_a["streak"]} 天 |'
          f' 防守點: ${slot_a["stop_loss"]:.2f}\n'
      )
    else:
      report += '👑 [Slot A 衛冕者] 目前空缺\n'

    if slot_b:
      s = slot_b['symbol']
      name = tw_stocks.get(f'{s}.TW', tw_stocks.get(f'{s}.TWO', ''))
      is_high = is_5y_high_map.get(s, False)
      tag = ' | 突破5年新高: Yes' if is_high else ' | 突破5年新高: No'
      report += (
          f'⚡ [Slot B 挑戰者] {s} {name}{tag}\n   能量: {slot_b["score"]}% |'
          f' 建議防守: ${slot_b["stop_loss"]:.2f}\n'
      )
    else:
      report += '⚡ [Slot B 挑戰者] 無標的跨越 90 分發動線\n'

    send_email_report(report)
