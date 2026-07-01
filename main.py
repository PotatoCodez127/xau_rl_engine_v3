import os
import time
import asyncio
import logging
import sys
import requests
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
from collections import deque
from fastapi import FastAPI, Request, Response
from dotenv import load_dotenv

# Internal V3 Imports
from m1_live_simulator import StreamingFeatureEngine
from wa_manager import WhatsAppCopilot
# Inject the new ONNX Engine (Replaces PyTorch and SB3 imports)
from onnx_inference import LocalInferenceEngine

load_dotenv()

# ==============================================================
# 0. LOGGING CONFIGURATION (Dual-Tier: File + Console)
# ==============================================================
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("XAU_Live_Engine")
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S UTC')

file_handler = logging.FileHandler("logs/live_engine.log", encoding='utf-8')
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- CONFIGURATION ---
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "PHONE_NUMBER_ID_NOT_SET")
WHATSAPP_API_URL = os.getenv("WHATSAPP_API_URL", f"https://graph.facebook.com/v17.0/{PHONE_NUMBER_ID}/messages")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "YOUR_ACCESS_TOKEN")
HEADERS = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}

# Point to your newly exported ONNX graphs (from Colab)
ORACLE_ONNX_PATH = "models/deployed/oracle_v3.onnx"
MANAGER_ONNX_PATH = "models/deployed/manager_actor_v3.onnx"

app = FastAPI(title="XAU Quant Copilot - Live Engine")
wa_manager = WhatsAppCopilot()

# --- CORE SYSTEMS ---
class LiveMT5Feed:
    """Asynchronous background connector to the open MT5 terminal."""
    def __init__(self, xau_symbol="XAUUSDr", dxy_symbol="USDIndex"):
        if not mt5.initialize():
            raise ConnectionError(f"MT5 Initialization failed: {mt5.last_error()}")
        self.xau = xau_symbol
        self.dxy = dxy_symbol
        logger.info(f"[MT5] Terminal Connected. Streaming {self.xau} & {self.dxy}")

    def get_latest_tick(self):
        tick_xau = mt5.symbol_info_tick(self.xau)
        if tick_xau is None:
            logger.warning("[MT5] Connection lost. Attempting to re-initialize...")
            mt5.initialize()
            return None, None
            
        tick_dxy = mt5.symbol_info_tick(self.dxy)
        if tick_dxy is None:
            return None, None

        timestamp = pd.to_datetime(tick_xau.time, unit='s').tz_localize('UTC')
        tick_row = {
            "xau_open": tick_xau.bid,
            "xau_high": tick_xau.ask,
            "xau_low": tick_xau.bid,
            "xau_close": tick_xau.bid,
            "dxy_close": tick_dxy.bid
        }
        return timestamp, tick_row

    def fetch_historical_warmup(self, bars=16000):
        logger.info(f"[MT5] Fetching last {bars} M1 bars for Instant Warmup...")
        rates_xau = mt5.copy_rates_from_pos(self.xau, mt5.TIMEFRAME_M1, 0, bars)
        rates_dxy = mt5.copy_rates_from_pos(self.dxy, mt5.TIMEFRAME_M1, 0, bars)
        
        if rates_xau is None or rates_dxy is None:
            raise ValueError("Failed to fetch historical data. Check MT5 connection.")
            
        df_xau = pd.DataFrame(rates_xau)
        df_xau['datetime'] = pd.to_datetime(df_xau['time'], unit='s').dt.tz_localize('UTC')
        df_xau.set_index('datetime', inplace=True)
        df_xau.rename(columns={'open': 'xau_open', 'high': 'xau_high', 'low': 'xau_low', 'close': 'xau_close'}, inplace=True)
        
        df_dxy = pd.DataFrame(rates_dxy)
        df_dxy['datetime'] = pd.to_datetime(df_dxy['time'], unit='s').dt.tz_localize('UTC')
        df_dxy.set_index('datetime', inplace=True)
        df_dxy.rename(columns={'close': 'dxy_close'}, inplace=True)
        
        master_df = df_xau[['xau_open', 'xau_high', 'xau_low', 'xau_close']].join(df_dxy[['dxy_close']], how='inner').dropna()
        return master_df


feed = LiveMT5Feed()
feature_engine = StreamingFeatureEngine(window_size=1000)

# Initialize the lightweight ONNX Engine
if os.path.exists(ORACLE_ONNX_PATH) and os.path.exists(MANAGER_ONNX_PATH):
    inference_engine = LocalInferenceEngine(ORACLE_ONNX_PATH, MANAGER_ONNX_PATH)
else:
    inference_engine = None
    logger.warning("⚠️ ONNX weights not found. Signals will not trigger. Awaiting weights from Colab.")

feature_buffer = deque(maxlen=30)
feature_cols = [
    "h4_trend", "h1_vol_regime", "close_frac_diff", "mom_1_norm", "mom_4_norm", "dxy_pct_change_15m",
    "dist_ema_50_norm", "dist_rolling_max_15m_norm", "dist_rolling_min_15m_norm",
    "dist_res_zone_top_15m_norm", "dist_res_zone_bottom_15m_norm", "dist_sup_zone_top_15m_norm", "dist_sup_zone_bottom_15m_norm",
    "dist_res_zone_top_30m_norm", "dist_res_zone_bottom_30m_norm", "dist_sup_zone_top_30m_norm", "dist_sup_zone_bottom_30m_norm",
    "dist_res_zone_top_4h_norm", "dist_res_zone_bottom_4h_norm", "dist_sup_zone_top_4h_norm", "dist_sup_zone_bottom_4h_norm",
    "dist_daily_eq_norm", "dist_pivot_norm", "dist_R1_norm", "dist_S1_norm"
]

@app.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    if mode == "subscribe" and token == os.getenv("VERIFY_TOKEN"):
        return int(challenge)
    return Response(status_code=403)

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    try:
        data = await request.json()
        messages = data.get('entry', [{}])[0].get('changes', [{}])[0].get('value', {}).get('messages', [])
        if messages:
            phone_number = messages[0].get('from')
            wa_manager.update_interaction(phone_number)
    except Exception as e:
        logger.error(f"[Webhook Error] Malformed payload: {e}")
    return {"status": "ok"}

async def live_trading_loop():
    logger.info("🚀 Initiating Live CPU Inference Loop. Standing by for ticks...")
    
    bars_since_last_trade = 0
    trades_today = 0
    current_day = datetime.now(timezone.utc).date()
    
    simulated_equity_ratio = 1.0  
    simulated_drawdown_ratio = 0.0

    logger.info("⏳ Executing Instant Historical Warmup Sequence...")
    try:
        hist_df = feed.fetch_historical_warmup(bars=15000)
        for timestamp, row in hist_df.iterrows():
            tick_row_dict = row.to_dict()
            latest_15m_features = feature_engine.process_m1_tick(timestamp, tick_row_dict)
            if latest_15m_features is not None:
                feature_vector = [latest_15m_features.get(c, 0.0) if not np.isnan(latest_15m_features.get(c, 0.0)) else 0.0 for c in feature_cols]
                feature_buffer.append(feature_vector)
        logger.info(f"✅ Warmup Complete. Engine Memory Saturated.")
    except Exception as e:
        logger.error(f"⚠️ Warmup Failed: {e}.")

    while True:
        timestamp, tick_row = feed.get_latest_tick()
        
        if timestamp and tick_row:
            if timestamp.date() > current_day:
                logger.info(f"[{timestamp}] 📅 New trading day detected (UTC). Daily execution limit reset to 0.")
                current_day = timestamp.date()
                trades_today = 0

            latest_15m_features = feature_engine.process_m1_tick(timestamp, tick_row)
            
            if latest_15m_features is not None:
                bars_since_last_trade += 1
                feature_vector = [latest_15m_features.get(c, 0.0) if not np.isnan(latest_15m_features.get(c, 0.0)) else 0.0 for c in feature_cols]
                feature_buffer.append(feature_vector)

                if len(feature_buffer) == 30 and inference_engine is not None:
                    # --- 1. PHASE A: ORACLE (CPU ONNX) ---
                    buffer_array = np.array(feature_buffer)
                    prob_hold, prob_long, prob_short = inference_engine.predict_oracle(buffer_array)

                    EXECUTION_THRESHOLD = 0.35
                    current_h4_trend = latest_15m_features.get("h4_trend", 0.0)
                    env_atr = latest_15m_features.get("env_atr", 1.0)
                    
                    logger.info(f"[{timestamp}] 📊 15m Closed | Trend (H4): {current_h4_trend:.4f} | ATR: {env_atr:.2f} | Probs -> H: {prob_hold:.3f} | L: {prob_long:.3f} | S: {prob_short:.3f}")

                    # --- 2. MASTER-SLAVE TRIGGER ---
                    direction = 0
                    if prob_long > EXECUTION_THRESHOLD and prob_long > prob_hold and current_h4_trend > 0:
                        direction = 1
                    elif prob_short > EXECUTION_THRESHOLD and prob_short > prob_hold and current_h4_trend < 0:
                        direction = 2

                    if direction != 0:
                        if bars_since_last_trade < 4:
                            logger.info(f"[{timestamp}] ⏳ High-conviction signal blocked by 1h cooldown.")
                            direction = 0  
                        elif trades_today >= 1:
                            logger.info(f"[{timestamp}] 🛑 High-conviction signal blocked by Daily Limit.")
                            direction = 0

                    if direction != 0:
                        # --- 3. PHASE B: SAC MANAGER (CPU ONNX) ---
                        obs = np.zeros(31, dtype=np.float32)
                        obs[:25] = feature_vector
                        obs[25], obs[26], obs[27] = prob_hold, prob_long, prob_short
                        obs[28] = simulated_equity_ratio
                        obs[29] = simulated_drawdown_ratio
                        obs[30] = float(np.clip(bars_since_last_trade / 480.0, 0.0, 1.0))
                        
                        raw_sl, raw_tp = inference_engine.predict_manager(obs)
                        
                        # Apply Mathematical Asymmetry Bounds
                        sl_mult = 0.5 + ((raw_sl + 1.0) * (2.0 - 0.5)) / 2.0
                        tp_mult = 1.0 + ((raw_tp + 1.0) * (3.0 - 1.0)) / 2.0
                        tp_mult_used = sl_mult * tp_mult 
                        
                        entry_price = tick_row["xau_close"]
                        sl_distance = (env_atr * sl_mult)
                        tp_distance = (env_atr * tp_mult_used)
                        
                        signal_data = {
                            "type": "Long" if direction == 1 else "Short",
                            "entry": entry_price,
                            "sl": entry_price - sl_distance if direction == 1 else entry_price + sl_distance,
                            "tp": entry_price + tp_distance if direction == 1 else entry_price - tp_distance,
                            "risk_profile": "Standard WFA Config"
                        }
                        
                        logger.info(f"[{timestamp}] 🚀 VALID SIGNAL DETECTED. Dispatching to WhatsApp...")
                        wa_manager.broadcast_signal(signal_data)
                        
                        bars_since_last_trade = 0
                        trades_today += 1

        await asyncio.sleep(0.25)

async def reminder_loop():
    while True:
        wa_manager.check_and_send_reminders()
        await asyncio.sleep(300)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(live_trading_loop())
    asyncio.create_task(reminder_loop())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 