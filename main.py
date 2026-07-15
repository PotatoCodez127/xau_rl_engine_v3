import os
import json
import time
import asyncio
import logging
import sys
import numpy as np
import pandas as pd
import onnxruntime as ort
import MetaTrader5 as mt5
from datetime import datetime, timezone
from collections import deque
from fastapi import FastAPI, Request, Response
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from m1_live_simulator import StreamingFeatureEngine
from wa_manager import WhatsAppCopilot

load_dotenv()

# ==============================================================
# 0. LOGGING CONFIGURATION
# ==============================================================
os.makedirs("logs", exist_ok=True)

ort.set_default_logger_severity(3)

logger = logging.getLogger("XAU_Live_Engine")
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S UTC')

file_handler = logging.FileHandler("logs/live_engine.log", encoding='utf-8')
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# --- PATHS ---
ORACLE_ONNX_PATH = "models/deployed/oracle_v3.onnx"
MANAGER_ONNX_PATH = "models/deployed/manager_actor_v3.onnx"
STATE_FILE = "logs/engine_state.json"
NEURAL_LOG_PATH = "logs/neural_research_log_live.csv"
JOURNAL_PATH = "logs/high_fidelity_journal_live.csv"
MAX_TRADES_PER_DAY = 5

wa_manager = WhatsAppCopilot()

def numpy_softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# ==============================================================
# 1. LIVE ENVIRONMENT STATE (Mimics XAUDynamicEnv exactly)
# ==============================================================
class LiveEnvState:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self._load()
        self.trade_history = deque(self.data.get("trade_history", []), maxlen=20)

    def _load(self):
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("Initializing new Live Env state.")
            return {
                "current_day": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "trades_today": 0,
                "bars_since_last_trade": 4,
                "balance": 10000.0,
                "peak_balance": 10000.0,
                "trade_history": []
            }

    def save(self):
        self.data["trade_history"] = list(self.trade_history)
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f)
            
    def update_daily_limits(self, current_dt):
        today_str = current_dt.strftime("%Y-%m-%d")
        if today_str > self.data["current_day"]:
            logger.info(f"📅 New trading day ({today_str}). Limits reset.")
            self.data["current_day"] = today_str
            self.data["trades_today"] = 0
            self.save()

state = LiveEnvState(STATE_FILE)

# ==============================================================
# 2. MT5 ASYNC FEED
# ==============================================================
class LiveMT5Feed:
    def __init__(self, xau_symbol="XAUUSDr", dxy_symbol="USDIndex"):
        if not mt5.initialize():
            logger.error(f"[MT5] Init failed: {mt5.last_error()}")
        self.xau = xau_symbol
        self.dxy = dxy_symbol
        logger.info(f"[MT5] Connected. Streaming {self.xau} & {self.dxy}")

    def get_latest_tick(self):
        tick_xau = mt5.symbol_info_tick(self.xau)
        tick_dxy = mt5.symbol_info_tick(self.dxy)
        
        if tick_xau is None or tick_dxy is None:
            mt5.initialize()
            return None, None

        timestamp = pd.to_datetime(tick_xau.time, unit='s').tz_localize('UTC')
        return timestamp, {
            "xau_open": tick_xau.bid,
            "xau_high": tick_xau.ask,
            "xau_low": tick_xau.bid,
            "xau_close": tick_xau.bid,
            "dxy_close": tick_dxy.bid
        }

    def fetch_historical_warmup(self, bars=16000):
        rates_xau = mt5.copy_rates_from_pos(self.xau, mt5.TIMEFRAME_M1, 0, bars)
        rates_dxy = mt5.copy_rates_from_pos(self.dxy, mt5.TIMEFRAME_M1, 0, bars)
        
        # 2. Add validation block to capture connection/symbol issues cleanly
        if rates_xau is None or rates_dxy is None:
            raise ValueError(
                f"Failed to fetch historical rates. Ensure MT5 terminal is running, "
                f"logged into your broker, and that symbols '{self.xau}' and '{self.dxy}' "
                f"are visible in your Market Watch window."
            )
        
        df_xau = pd.DataFrame(rates_xau)
        df_xau['datetime'] = pd.to_datetime(df_xau['time'], unit='s').dt.tz_localize('UTC')
        df_xau.set_index('datetime', inplace=True)
        df_xau.rename(columns={'open': 'xau_open', 'high': 'xau_high', 'low': 'xau_low', 'close': 'xau_close'}, inplace=True)
        
        df_dxy = pd.DataFrame(rates_dxy)
        df_dxy['datetime'] = pd.to_datetime(df_dxy['time'], unit='s').dt.tz_localize('UTC')
        df_dxy.set_index('datetime', inplace=True)
        df_dxy.rename(columns={'close': 'dxy_close'}, inplace=True)
        
        return df_xau[['xau_open', 'xau_high', 'xau_low', 'xau_close']].join(df_dxy[['dxy_close']], how='inner').dropna()

feed = LiveMT5Feed()
feature_engine = StreamingFeatureEngine(window_size=1000)
feature_buffer = deque(maxlen=30)

feature_cols = [
    "h4_trend", "rsi_14_norm", "close_frac_diff_norm", "dxy_pct_change_15m_norm",
    "mom_1_norm", "mom_4_norm", "h1_vol_regime_norm", "dist_ema_50_norm",
    "dist_rolling_max_15m_norm", "dist_rolling_min_15m_norm",
    "dist_res_zone_top_15m_norm", "dist_res_zone_bottom_15m_norm", "dist_sup_zone_top_15m_norm", "dist_sup_zone_bottom_15m_norm",
    "dist_res_zone_top_30m_norm", "dist_res_zone_bottom_30m_norm", "dist_sup_zone_top_30m_norm", "dist_sup_zone_bottom_30m_norm",
    "dist_res_zone_top_4h_norm", "dist_res_zone_bottom_4h_norm", "dist_sup_zone_top_4h_norm", "dist_sup_zone_bottom_4h_norm",
    "dist_daily_eq_norm", "dist_pivot_norm", "dist_R1_norm", "dist_S1_norm",
    "prob_long", "prob_short", "prob_hold"
]

oracle_session = None
manager_session = None

# ==============================================================
# 3. CORE ASYNC ENGINE LOOP
# ==============================================================
def append_csv_log(filepath, data_dict):
    df = pd.DataFrame([data_dict])
    if not os.path.isfile(filepath):
        df.to_csv(filepath, index=False)
    else:
        df.to_csv(filepath, mode='a', header=False, index=False)

async def live_trading_loop():
    logger.info("🚀 Initiating Deep Diagnostics Parity Live Loop...")

    try:
        hist_df = await asyncio.to_thread(feed.fetch_historical_warmup, 15000)
        for timestamp, row in hist_df.iterrows():
            features = feature_engine.process_m1_tick(timestamp, row.to_dict())
            if features is not None:
                feature_buffer.append([features.get(c, 0.0) for c in feature_cols])
        logger.info("✅ Engine Memory Saturated.")
    except Exception as e:
        logger.error(f"⚠️ Warmup Failed: {e}")

    oracle_input_name = oracle_session.get_inputs()[0].name
    manager_input_name = manager_session.get_inputs()[0].name

    while True:
        try:
            timestamp, tick_row = await asyncio.to_thread(feed.get_latest_tick)
            
            if timestamp and tick_row:
                state.update_daily_limits(timestamp)
                latest_features = feature_engine.process_m1_tick(timestamp, tick_row)
                
                if latest_features is not None:
                    state.data["bars_since_last_trade"] += 1
                    feature_vector = [latest_features.get(c, 0.0) for c in feature_cols]
                    feature_buffer.append(feature_vector)

                    if len(feature_buffer) == 30:
                        # --- PHASE A: CONTINUOUS ORACLE INFERENCE ---
                        buffer_array = np.array(feature_buffer, dtype=np.float32).reshape(1, 30, 29)
                        logits = await asyncio.to_thread(
                            oracle_session.run, None, {oracle_input_name: buffer_array}
                        )
                        probs = numpy_softmax(logits[0])[0]
                        prob_hold, prob_long, prob_short = probs[0], probs[1], probs[2]

                        # Re-inject live probabilities back into the feature vector for the Manager
                        feature_vector[26] = prob_long
                        feature_vector[27] = prob_short
                        feature_vector[28] = prob_hold

                        # Ensure MT5 Live Balance Parity with XAUDynamicEnv
                        account = await asyncio.to_thread(mt5.account_info)
                        if account:
                            current_bal = account.balance
                            state.data["peak_balance"] = max(state.data["peak_balance"], current_bal)
                        else:
                            current_bal = state.data["balance"]

                        # --- PHASE B: CONTINUOUS MANAGER INFERENCE (Uncapped) ---
                        obs = np.zeros(32, dtype=np.float32) # Increased to 32 dimensions
                        obs[:29] = feature_vector
                        obs[29] = float(np.clip(current_bal / 10000.0, 0.0, 10.0))
                        obs[30] = float(np.clip((state.data["peak_balance"] - current_bal) / state.data["peak_balance"], 0.0, 1.0))
                        obs[31] = float(np.clip(state.data["bars_since_last_trade"] / 480.0, 0.0, 1.0))
                        
                        onnx_obs = obs.reshape(1, -1)
                        action = await asyncio.to_thread(
                            manager_session.run, None, {manager_input_name: onnx_obs}
                        )
                        raw_sl, raw_tp = action[0][0]

                        # Action Scaling (Symmetry with XAUDynamicEnv)
                        sl_mult_used = 0.5 + ((raw_sl + 1.0) * (2.0 - 0.5)) / 2.0
                        tp_mult_ratio = 1.0 + ((raw_tp + 1.0) * (3.0 - 1.0)) / 2.0
                        tp_mult_used = sl_mult_used * tp_mult_ratio 

                        # Calculate Imbalance Ratio for Heartbeat
                        long_count = sum(1 for d in state.trade_history if d == 1)
                        short_count = sum(1 for d in state.trade_history if d == 2)
                        imbalance_ratio = max(long_count, short_count) / max(1, len(state.trade_history))

                        # Dual-Tier Logging: Neural Heartbeat
                        append_csv_log(NEURAL_LOG_PATH, {
                            "datetime": str(timestamp),
                            "prob_hold": prob_hold,
                            "prob_long": prob_long,
                            "prob_short": prob_short,
                            "sl_mult_intent": sl_mult_used,
                            "tp_mult_intent": tp_mult_used,
                            "imbalance_ratio": imbalance_ratio,
                            "step_reward": 0.0 # Reward is arbitrary in forward live feed
                        })

                        # --- MASTER SLAVE TRIGGER (Symmetry with XAUDynamicEnv) ---
                        EXECUTION_THRESHOLD = 0.40
                        # --- CONSOLE HEARTBEAT LOGGER ---
                        current_h4_trend = latest_features.get("h4_trend", 0.0)
                        
                        # Determine if we are currently blocked by environmental constraints
                        cooldown_blocked = state.data["bars_since_last_trade"] < 4
                        limit_blocked = state.data["trades_today"] >= MAX_TRADES_PER_DAY  # Ensure this variable matches your settings
                        
                        status_flag = "⏸️ COOLDOWN" if cooldown_blocked else ("🚫 LIMIT EXCEEDED" if limit_blocked else "🟢 READY")
                        trend_direction = "UP" if current_h4_trend > 0 else "DOWN"

                        logger.info(
                            f"💓 HEARTBEAT [{timestamp.strftime('%H:%M:%S')}] | "
                            f"Status: {status_flag:11} | "
                            f"H4 Trend: {trend_direction} ({current_h4_trend:+.2f}) | "
                            f"P_Hold: {prob_hold:.2%}, P_Long: {prob_long:.2%}, P_Short: {prob_short:.2%} | "
                            f"SL Mult: {sl_mult_used:.2f}, TP Mult: {tp_mult_used:.2f} | "
                            f"Bars Since: {state.data['bars_since_last_trade']}/4"
                        )
                        env_atr = latest_features.get("env_atr", 1.0)

                        direction = 0
                        if prob_long > EXECUTION_THRESHOLD and prob_long > prob_hold and current_h4_trend > 0:
                            direction = 1
                        elif prob_short > EXECUTION_THRESHOLD and prob_short > prob_hold and current_h4_trend < 0:
                            direction = 2

                        # Environment Caps Check
                        if direction != 0:
                            if state.data["bars_since_last_trade"] < 4 or state.data["trades_today"] >= MAX_TRADES_PER_DAY:
                                direction = 0

                        # --- EXECUTION DISPATCH ---
                        if direction != 0:
                            entry_price = tick_row["xau_close"]
                            
                            signal_data = {
                                "type": "Long" if direction == 1 else "Short",
                                "entry": entry_price,
                                "sl": entry_price - (env_atr * sl_mult_used) if direction == 1 else entry_price + (env_atr * sl_mult_used),
                                "tp": entry_price + (env_atr * tp_mult_used) if direction == 1 else entry_price - (env_atr * tp_mult_used),
                                "risk_profile": f"SL Mult: {sl_mult_used:.2f} | TP Mult: {tp_mult_used:.2f}"
                            }
                            
                            logger.info(f"[{timestamp}] 🚀 VALID SIGNAL. Dispatching via WA...")
                            await asyncio.to_thread(wa_manager.broadcast_signal, signal_data)
                            
                            # Dual-Tier Logging: Trade Journal
                            append_csv_log(JOURNAL_PATH, {
                                "datetime": str(timestamp),
                                "prob_long": prob_long,
                                "prob_short": prob_short,
                                "sl_mult_used": sl_mult_used,
                                "tp_mult_used": tp_mult_used,
                                "simulated_pnl": 0.0, # Real PnL evaluated next step via MT5 balance
                                "account_balance": current_bal
                            })

                            # Update Environmental State
                            state.trade_history.append(direction)
                            state.data["bars_since_last_trade"] = 0
                            state.data["trades_today"] += 1
                            state.data["balance"] = current_bal
                            state.save()

        except Exception as e:
            logger.error(f"Live Loop Exception: {e}", exc_info=True)
            
        await asyncio.sleep(0.5) 

async def reminder_loop():
    while True:
        try:
            await asyncio.to_thread(wa_manager.check_and_send_reminders)
        except Exception as e:
            logger.error(f"Reminder Loop Error: {e}")
        await asyncio.sleep(300)

# ==============================================================
# 4. FASTAPI LIFESPAN & ROUTING
# ==============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global oracle_session, manager_session
    if os.path.exists(ORACLE_ONNX_PATH) and os.path.exists(MANAGER_ONNX_PATH):
        oracle_session = ort.InferenceSession(ORACLE_ONNX_PATH, providers=['CPUExecutionProvider'])
        manager_session = ort.InferenceSession(MANAGER_ONNX_PATH, providers=['CPUExecutionProvider'])
    else:
        logger.error("🚨 ONNX Models not found. Engine offline.")
        
    trading_task = asyncio.create_task(live_trading_loop())
    reminder_task = asyncio.create_task(reminder_loop())
    
    yield 
    
    trading_task.cancel()
    reminder_task.cancel()
    mt5.shutdown()
    state.save()

app = FastAPI(title="XAU Quant Copilot - Live Engine", lifespan=lifespan)

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
            await asyncio.to_thread(wa_manager.update_interaction, phone_number)
    except Exception as e:
        logger.error(f"[Webhook Error] Malformed payload: {e}")
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)