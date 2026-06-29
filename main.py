import os
import time
import asyncio
import requests
import torch
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
from collections import deque
from fastapi import FastAPI, Request
from stable_baselines3 import SAC

# Internal V3 Imports
from m1_live_simulator import StreamingFeatureEngine
from models.oracle.attention_net import TemporalAttentionOracle

# --- CONFIGURATION ---
WHATSAPP_API_URL = os.getenv("WHATSAPP_API_URL", "https://graph.facebook.com/v17.0/YOUR_PHONE_NUMBER_ID/messages")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "YOUR_ACCESS_TOKEN")
HEADERS = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}

ORACLE_WEIGHTS = "models/oracle/best_oracle.pth"
MANAGER_WEIGHTS = "models/manager/saved/wfa_43/best_model.zip"  # Update to latest WFA split

# --- CORE SYSTEMS ---

class WhatsAppCopilot:
    """Manages the 24-hour interaction window and broadcast logic for live signals."""
    def __init__(self):
        self.subscribers = {}  # Format: { "phone_number": "last_interaction_utc_timestamp" }
        self.window_limit = timedelta(hours=24)
        self.reminder_threshold = timedelta(hours=23, minutes=30)

    def update_interaction(self, phone_number: str):
        """Called via FastAPI webhook when a user sends a message, refreshing the 24h window."""
        self.subscribers[phone_number] = datetime.now(timezone.utc)
        print(f"[WhatsApp] Window refreshed for {phone_number}. Valid for 24h.")
        self._send_message(phone_number, "System Synced. You are actively receiving XAU Live Signals.")

    def broadcast_signal(self, signal_data: dict):
        """Sends the generated trade matrix to all users within the open window."""
        now = datetime.now(timezone.utc)
        active_users = [
            num for num, last_time in self.subscribers.items()
            if now - last_time < self.window_limit
        ]

        message_body = (
            f"⚜️ XAU_RL_V3 LIVE SIGNAL ⚜️\n\n"
            f"Action: {signal_data['type']}\n"
            f"Entry: {signal_data['entry']:.3f}\n"
            f"Stop Loss: {signal_data['sl']:.3f}\n"
            f"Take Profit: {signal_data['tp']:.3f}\n\n"
            f"Calculated Risk: {signal_data['risk_profile']}"
        )

        for user in active_users:
            self._send_message(user, message_body)
            print(f"[WhatsApp] Signal dispatched to {user}")

    def check_and_send_reminders(self):
        """Warns users their receiving window is about to mathematically expire."""
        now = datetime.now(timezone.utc)
        for num, last_time in self.subscribers.items():
            elapsed = now - last_time
            if self.reminder_threshold <= elapsed < self.window_limit:
                reminder = "⚠️ Your signal window is closing in less than 30 minutes! Reply 'SYNC' to keep the live feed open."
                self._send_message(num, reminder)
                print(f"[WhatsApp] Reminder dispatched to {num}")

    def _send_message(self, to_number: str, text: str):
        payload = {
            "messaging_product": "whatsapp",
            "to": to_number,
            "type": "text",
            "text": {"body": text}
        }
        try:
            requests.post(WHATSAPP_API_URL, headers=HEADERS, json=payload)
        except Exception as e:
            print(f"[WhatsApp] Failed to send message: {e}")


class LiveMT5Feed:
    """Asynchronous background connector to the open MT5 terminal."""
    def __init__(self, xau_symbol="XAUUSD", dxy_symbol="DXY"):
        if not mt5.initialize():
            raise ConnectionError(f"MT5 Initialization failed: {mt5.last_error()}")
        self.xau = xau_symbol
        self.dxy = dxy_symbol
        print(f"[MT5] Terminal Connected. Streaming {self.xau} & {self.dxy}")

    def get_latest_tick(self):
        tick_xau = mt5.symbol_info_tick(self.xau)
        tick_dxy = mt5.symbol_info_tick(self.dxy)

        if tick_xau is None or tick_dxy is None:
            return None, None

        # Enforce UTC temporal synchronization for time-series parity
        timestamp = pd.to_datetime(tick_xau.time, unit='s').tz_localize('UTC')
        
        tick_row = {
            "xau_open": tick_xau.bid,
            "xau_high": tick_xau.ask,
            "xau_low": tick_xau.bid,
            "xau_close": tick_xau.bid,
            "dxy_close": tick_dxy.bid
        }
        return timestamp, tick_row


# --- FASTAPI APP INITIALIZATION ---
app = FastAPI(title="XAU Quant Copilot - Live Engine")
wa_manager = WhatsAppCopilot()
feed = LiveMT5Feed()
feature_engine = StreamingFeatureEngine(window_size=1000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_buffer = deque(maxlen=30)
feature_cols = [
    "h4_trend", "h1_vol_regime", "close_frac_diff", "mom_1_norm", "mom_4_norm", "dxy_pct_change_15m",
    "dist_ema_50_norm", "dist_rolling_max_15m_norm", "dist_rolling_min_15m_norm",
    "dist_res_zone_top_15m_norm", "dist_res_zone_bottom_15m_norm", "dist_sup_zone_top_15m_norm", "dist_sup_zone_bottom_15m_norm",
    "dist_res_zone_top_30m_norm", "dist_res_zone_bottom_30m_norm", "dist_sup_zone_top_30m_norm", "dist_sup_zone_bottom_30m_norm",
    "dist_res_zone_top_4h_norm", "dist_res_zone_bottom_4h_norm", "dist_sup_zone_top_4h_norm", "dist_sup_zone_bottom_4h_norm",
    "dist_daily_eq_norm", "dist_pivot_norm", "dist_R1_norm", "dist_S1_norm"
]

# Load Neural Architecture
print("[Neural Core] Loading Phase A (Temporal Attention Oracle)...")
oracle = TemporalAttentionOracle(input_dim=len(feature_cols), seq_len=30).to(device)
if os.path.exists(ORACLE_WEIGHTS):
    oracle.load_state_dict(torch.load(ORACLE_WEIGHTS, map_location=device))
oracle.eval()

print("[Neural Core] Loading Phase B (SAC Manager)...")
if os.path.exists(MANAGER_WEIGHTS):
    manager = SAC.load(MANAGER_WEIGHTS, device=device)
else:
    manager = None
    print("⚠️ Warning: SAC Manager weights not found. Signals will not trigger.")


@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    """Intercepts Meta webhooks to reset client interaction windows."""
    try:
        data = await request.json()
        messages = data.get('entry', [{}])[0].get('changes', [{}])[0].get('value', {}).get('messages', [])
        if messages:
            phone_number = messages[0].get('from')
            # The client sent a message, refreshing the 24h gatekeeper window
            wa_manager.update_interaction(phone_number)
    except Exception as e:
        print(f"[Webhook Error] Malformed payload: {e}")
        
    return {"status": "ok"}


@app.get("/")
async def health_check():
    return {"status": "Live", "active_clients": len(wa_manager.subscribers)}


async def live_trading_loop():
    """Background Event Loop: Pulls ticks, evaluates 15m structural closures, triggers RL inference."""
    print("\n🚀 Initiating Live M1 Execution Loop. Standing by for ticks...")
    
    # State tracking for the manager network observations
    bars_since_last_trade = 0
    simulated_equity_ratio = 1.0  
    simulated_drawdown_ratio = 0.0

    while True:
        timestamp, tick_row = feed.get_latest_tick()
        
        if timestamp and tick_row:
            latest_15m_features = feature_engine.process_m1_tick(timestamp, tick_row)
            
            # If the engine returns data, a 15m candle has mathematically closed
            if latest_15m_features is not None:
                bars_since_last_trade += 1
                
                # Extract and format features for the network
                feature_vector = [latest_15m_features.get(c, 0.0) if not np.isnan(latest_15m_features.get(c, 0.0)) else 0.0 for c in feature_cols]
                feature_buffer.append(feature_vector)

                if len(feature_buffer) == 30 and manager is not None:
                    # --- INFERENCE PHASE ---
                    window_tensor = torch.FloatTensor(np.array(feature_buffer)).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        logits = oracle(window_tensor)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    prob_hold, prob_long, prob_short = probs[0], probs[1], probs[2]

                    EXECUTION_THRESHOLD = 0.35
                    current_h4_trend = latest_15m_features.get("h4_trend", 0)
                    direction = 0

                    if prob_long > EXECUTION_THRESHOLD and prob_long > prob_short and current_h4_trend > 0:
                        direction = 1
                    elif prob_short > EXECUTION_THRESHOLD and prob_short > prob_long and current_h4_trend < 0:
                        direction = 2

                    if direction != 0:
                        # Construct Manager Observation Space
                        obs = np.zeros(31, dtype=np.float32)
                        obs[:25] = feature_vector
                        obs[25], obs[26], obs[27] = prob_hold, prob_long, prob_short
                        obs[28] = simulated_equity_ratio
                        obs[29] = simulated_drawdown_ratio
                        obs[30] = float(np.clip(bars_since_last_trade / 480.0, 0.0, 1.0))
                        
                        action, _ = manager.predict(obs, deterministic=True)
                        size_val, tp_val, sl_val = action[1], action[2], action[3]
                        
                        # De-normalize SL/TP metrics based on current environmental ATR
                        sl_mult = ((sl_val + 1.0) / 2.0) * 1.0 + 0.5
                        tp_mult = sl_mult * (((tp_val + 1.0) / 2.0) * 2.0 + 1.0)
                        env_atr = latest_15m_features.get("env_atr", 1.0)
                        
                        entry_price = tick_row["xau_close"]
                        sl_distance = (env_atr * sl_mult)
                        tp_distance = (env_atr * tp_mult)
                        
                        signal_data = {
                            "type": "Long" if direction == 1 else "Short",
                            "entry": entry_price,
                            "sl": entry_price - sl_distance if direction == 1 else entry_price + sl_distance,
                            "tp": entry_price + tp_distance if direction == 1 else entry_price - tp_distance,
                            "risk_profile": "Standard WFA Config"
                        }
                        
                        wa_manager.broadcast_signal(signal_data)
                        bars_since_last_trade = 0

        # Poll MT5 at 1Hz 
        await asyncio.sleep(1)


async def reminder_loop():
    """Background task running every 5 minutes to sweep and manage expiring client windows."""
    while True:
        wa_manager.check_and_send_reminders()
        await asyncio.sleep(300)


@app.on_event("startup")
async def startup_event():
    # Deploy asynchronous workers upon API spin-up
    asyncio.create_task(live_trading_loop())
    asyncio.create_task(reminder_loop())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)