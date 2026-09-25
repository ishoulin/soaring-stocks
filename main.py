import datetime
import json
import os
import re
import smtplib
import sys
from email.header import Header
from email.mime.text import MIMEText
import numpy as np
import pandas as pd
import requests
import time
import yfinance as yf

def fetch_data_safely(tickers, period="5y", batch_size=10, sleep_sec=1.5):
    """
    分批下載歷史資料，加入 request 間隔防止被 Yahoo 丟包卡死
    """
    all_data = pd.DataFrame()
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    
    print(f"總共 {len(tickers)} 檔股票，分為 {total_batches} 批次進行下載...")

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        current_batch = i // batch_size + 1
        print(f"[{current_batch}/{total_batches}] 正在下載: {', '.join(batch)}...")
        
        success = False
        for attempt in range(3):  # 最多嘗試 3 次
            try:
                # 關鍵設定：timeout=10, threads=False
                df = yf.download(
                    tickers=batch, 
                    period=period, 
                    group_by='ticker', 
                    threads=False, 
                    timeout=10,
                    progress=False
                )
                if not df.empty:
                    # 簡單合併處理
                    all_data = pd.concat([all_data, df], axis=1)
                    success = True
                    break
            except Exception as e:
                print(f"   ⚠️ 批次下載失敗 (第 {attempt+1} 次重試): {e}")
                time.sleep(2)
        
        if not success:
            print(f"   ❌ 該批次多次失敗，已自動跳過，避免阻塞流程。")
            
        # 禮貌性停頓，避免被 Yahoo 認定為 Bot 攻擊
        time.sleep(sleep_sec)

    return all_data

STATE_FILE = "zen_state.json"


# ====================================================
# 1. 核心評分引擎 (ZenMomentum Engine - 正宗築底版)
# ====================================================
class ZenMomentumEngine:

    def calculate_score(self, df):
        """正宗『強勢築底』評分引擎

        指標三維度：
        1. 築底位置 (40分): 剛拉開近20日低點 2%~8% 為黃金區；太高(追高)或太低(破底)扣分
        2. 籌碼壓縮 (35分): 近 10 日高低振幅越小，代表籌碼沉澱越完美
        3. 溫和點火 (25分): 5日均量略大於 20日均量 (1.1 ~ 1.5 倍最佳)，非無量亦非暴量
        """
        close = df["Close"].iloc[-1]
        volume = df["Volume"]

        low_20 = df["Low"].tail(20).min()

        # 1. 築底位置分 (40分)
        dist_from_low = (close - low_20) / (low_20 + 1e-6)
        if 0.02 <= dist_from_low <= 0.08:
            base_score = 40.0  # 剛完成打腳，最佳黃金築底區
        elif dist_from_low < 0.02:
            base_score = 25.0  # 太貼近低點，仍有再次破底風險
        else:
            base_score = max(0.0, 40.0 - (dist_from_low - 0.08) * 150)

        # 2. 籌碼壓縮分 (35分)：近 10 日振幅越小越好
        high_10 = df["High"].tail(10).max()
        low_10 = df["Low"].tail(10).min()
        range_10 = (high_10 - low_10) / (low_10 + 1e-6)
        squeeze_score = max(0.0, (1.0 - range_10 / 0.20)) * 35.0

        # 3. 溫和點火分 (25分)：5日均量 vs 20日均量
        v_5 = volume.tail(5).mean()
        v_20 = volume.tail(20).mean()
        v_ratio = v_5 / (v_20 + 1e-6)

        if 1.1 <= v_ratio <= 1.5:
            vol_score = 25.0  # 溫和增量，主力默默卡位
        elif v_ratio < 1.1:
            vol_score = (v_ratio / 1.1) * 18.0  # 量能太過沉悶
        else:
            vol_score = 20.0  # 暴量過頭

        total_score = round(base_score + squeeze_score + vol_score, 1)
        return min(total_score, 99.9)

    def load_state(self):
        """讀取歷史 Slot A 狀態以計算連霸天數"""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"last_slot_a_symbol": None, "streak": 0}

    def save_state(self, slot_a_symbol, streak):
        """儲存今日 Slot A 狀態"""
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {"last_slot_a_symbol": slot_a_symbol, "streak": streak}, f
                )
        except Exception as e:
            print(f"⚠️ 無法寫入狀態檔: {e}", flush=True)

    def run_daily_arena(self, stock_dict):
        candidates = []
        for symbol, df in stock_dict.items():
            try:
                score = self.calculate_score(df)
                close = round(float(df["Close"].iloc[-1]), 2)
                stop_loss = round(float(df["Low"].tail(20).min() * 0.98), 2)

                candidates.append({
                    "symbol": symbol,
                    "score": score,
                    "close": close,
                    "stop_loss": stop_loss,
                })
            except Exception:
                continue

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_5 = candidates[:5]

        history = self.load_state()
        slot_a = top_5[0] if len(top_5) > 0 else None

        if slot_a:
            if history.get("last_slot_a_symbol") == slot_a["symbol"]:
                slot_a["streak"] = history.get("streak", 0) + 1
            else:
                slot_a["streak"] = 1
            self.save_state(slot_a["symbol"], slot_a["streak"])

        slot_b = (
            top_5[1]
            if len(top_5) > 1 and top_5[1]["score"] >= 80.0
            else None
        )

        return slot_a, slot_b, top_5


# ====================================================
# 2. 抓取全台股清單 (嚴格排除 ETF / 權證 / TDR / REITs)
# ====================================================
def fetch_tw_stock_tickers():
    print("📡 正在從證交所抓取台股清單...", flush=True)
    stocks = {}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    urls = [
        ("https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", ".TW"),  # 上市
        ("https://isin.twse.com.tw/isin/C_public.jsp?strMode=4", ".TWO"),  # 上櫃
    ]

    exclude_keywords = [
        "ETF",
        "ETN",
        "認購",
        "認售",
        "牛證",
        "熊證",
        "展",
        "R1",
        "R2",
        "存託憑證",
        "特",
        "甲",
        "乙",
        "丙",
    ]

    for url, suffix in urls:
        try:
            res = requests.get(url, headers=headers, timeout=8)
            res.encoding = "big5"
            df_list = pd.read_html(res.text)
            if df_list:
                df = df_list[0]
                df.columns = df.iloc[0]
                df = df.iloc[1:]
                for entry in df["有價證券代號及名稱"].dropna():
                    parts = entry.split("\u3000")
                    if len(parts) == 2:
                        code, name = parts[0].strip(), parts[1].strip()

                        if not (len(code) == 4 and code.isdigit()):
                            continue

                        if any(kw in name for kw in exclude_keywords):
                            continue

                        stocks[f"{code}{suffix}"] = name
        except Exception as e:
            print(f"⚠️ 抓取 {suffix} 清單失敗: {e}", flush=True)

    print(f"✅ 成功獲取 {len(stocks)} 檔個股清單！", flush=True)
    return stocks


# ====================================================
# 3. Email 自動發送函式
# ====================================================
def send_email_report(report_text):
    sender = os.getenv("EMAIL_USER", "").strip()
    password = os.getenv("EMAIL_PASS", "").strip()
    receiver = os.getenv("EMAIL_RECEIVER", "").strip() or sender

    if not sender or not password:
        print(
            "⚠️ 未設定 Email 環境變數，跳過發信步驟，直接印出戰報：",
            flush=True,
        )
        print(report_text, flush=True)
        return

    msg = MIMEText(report_text, "plain", "utf-8")
    msg["Subject"] = Header(
        "⚡ ZenMomentum 每日台股強勢築底戰報", "utf-8"
    )
    msg["From"] = sender
    msg["To"] = receiver

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, [receiver], msg.as_string())
        print("📧 Email 戰報已成功發送！", flush=True)
    except Exception as e:
        print(f"❌ 發信失敗: {e}", flush=True)


# ====================================================
# 4. 主程式執行 (強效防卡死 Safe-Download)
# ====================================================
if __name__ == "__main__":
    tw_stocks = fetch_tw_stock_tickers()
    symbol_list = list(tw_stocks.keys())
    total_fetched_count = len(symbol_list)

    # 批次改小為 15，大幅降低被 Yahoo 封鎖或 Response 掛起的機率
    batch_size = 15
    stage1_passed_symbols = []

    print(
        "🚀 [第一階段] 快速掃描流動性（5日均量 >= 500張）...", flush=True
    )
    for i in range(0, total_fetched_count, batch_size):
        chunk = symbol_list[i : i + batch_size]
        print(
            f"  ⏳ 下載進度: {i}/{total_fetched_count}...",
            end="\r",
            flush=True,
        )

        try:
            # 關鍵防卡死：timeout=8, threads=False
            data = yf.download(
                chunk,
                period="6mo",
                group_by="ticker",
                threads=False,
                progress=False,
                timeout=8,
            )
            if data is None or data.empty:
                continue

            for symbol in chunk:
                try:
                    df = (
                        data[symbol].copy()
                        if isinstance(data.columns, pd.MultiIndex)
                        else data.copy()
                    )
                    df = df.dropna(subset=["Close"])
                    if not df.empty and len(df) >= 30:
                        if df["Volume"].tail(5).mean() >= 500000:
                            stage1_passed_symbols.append(symbol)
                except Exception:
                    continue
        except Exception:
            # 若該批次卡死或出錯，直接跳過該批次，絕不中斷整體執行
            continue

    print(
        f"\n✅ 第一階段完成：共 {len(stage1_passed_symbols)} 檔個股符合流動性條件",
        flush=True,
    )

    print(
        "🎯 [第二階段] 精準檢測「2年高點回落 <= 20%」與「突破 5 年新高」...",
        flush=True,
    )
    all_stock_data = {}
    is_5y_high_map = {}

    for i in range(0, len(stage1_passed_symbols), batch_size):
        chunk = stage1_passed_symbols[i : i + batch_size]
        try:
            data_5y = yf.download(
                chunk,
                period="5y",
                group_by="ticker",
                threads=False,
                progress=False,
                timeout=8,
            )
            if data_5y is None or data_5y.empty:
                continue

            for symbol in chunk:
                try:
                    df = (
                        data_5y[symbol].copy()
                        if isinstance(data_5y.columns, pd.MultiIndex)
                        else data_5y.copy()
                    )
                    df = df.dropna(subset=["Close"])

                    if not df.empty and len(df) >= 120:
                        high_2y = df["High"].tail(500).max()
                        current_close = df["Close"].iloc[-1]
                        drawdown_2y = (high_2y - current_close) / high_2y

                        if drawdown_2y <= 0.20:
                            high_5y_prev = df["High"].iloc[:-1].max()
                            is_breakout = current_close >= high_5y_prev

                            clean_symbol = symbol.split(".")[0]
                            all_stock_data[clean_symbol] = df
                            is_5y_high_map[clean_symbol] = is_breakout
                except Exception:
                    continue
        except Exception:
            continue

    valid_scanned_count = len(all_stock_data)
    print(
        f"🎉 篩選完成：最終共 {valid_scanned_count} 檔個股符合「高檔強勢築底」型態！",
        flush=True,
    )

    # --- 執行引擎與產生報告 ---
    engine = ZenMomentumEngine()
    slot_a, slot_b, top_5 = engine.run_daily_arena(all_stock_data)

    report = (
        f"📈 【ZenMomentum 盤後數據掃描】\n"
        f"• 全台股個股掃描總數：{total_fetched_count} 檔 (已排除 ETF/權證)\n"
        f"• 強勢築底合格標的：{valid_scanned_count} 檔（回落<20% + 均量>500張）\n"
        f"{'='*35}\n\n"
        f"📊 【盤後強勢 Top 5】\n"
        f"{'='*35}\n"
    )

    for rank, cand in enumerate(top_5, 1):
        s = cand["symbol"]
        name = tw_stocks.get(f"{s}.TW", tw_stocks.get(f"{s}.TWO", ""))
        is_high = is_5y_high_map.get(s, False)
        tag = " | 突破5年新高: Yes" if is_high else " | 突破5年新高: No"
        report += f"第 {rank} 名 | {s} {name} | 能量: {cand['score']}% | 收盤: ${cand['close']}{tag}\n"

    report += "\n🏆 【每日雙槽戰報】\n" + "=" * 35 + "\n"
    if slot_a:
        s = slot_a["symbol"]
        name = tw_stocks.get(f"{s}.TW", tw_stocks.get(f"{s}.TWO", ""))
        is_high = is_5y_high_map.get(s, False)
        tag = " | 突破5年新高: Yes" if is_high else " | 突破5年新高: No"
        report += (
            f"👑 [Slot A 衛冕者] {s} {name}{tag}\n"
            f"    連霸: {slot_a['streak']} 天 | 防守點: ${slot_a['stop_loss']:.2f}\n"
        )
    else:
        report += "👑 [Slot A 衛冕者] 目前空缺\n"

    if slot_b:
        s = slot_b["symbol"]
        name = tw_stocks.get(f"{s}.TW", tw_stocks.get(f"{s}.TWO", ""))
        is_high = is_5y_high_map.get(s, False)
        tag = " | 突破5年新高: Yes" if is_high else " | 突破5年新高: No"
        report += (
            f"⚡ [Slot B 挑戰者] {s} {name}{tag}\n"
            f"    能量: {slot_b['score']}% | 建議防守: ${slot_b['stop_loss']:.2f}\n"
        )
    else:
        report += "⚡ [Slot B 挑戰者] 無標的跨越 80 分發動線\n"

    send_email_report(report)
