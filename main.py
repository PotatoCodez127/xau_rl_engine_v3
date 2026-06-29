import asyncio
import os
import time
import torch
import numpy as np
from datetime import datetime, timezone
from collections import deque

from stable_baselines3 import SAC
from models.oracle.attention_net import TemporalAttentionOracle
from m1_live_simulator import StreamingFeatureEngine
from live_feed import LiveMT5Feed
from wa_manager import WhatsAppManager

async def live_trading_loop(wa_manager):
    print("\n🚀 Initiating Live M1 Execution Loop...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Core Components
    feed = LiveMT5Feed(xau_symbol="XAUUSD", dxy_symbol="DXY")
    feature_engine = StreamingFeatureEngine(window_size=1000)
    
    # 2. Load Neural Networks
    feature_cols = [
        "h4_trend", "h1_vol_regime", "close_frac_diff", "mom_1_norm", "mom_4_norm", "dxy_pct_change_15m",
        "dist_ema_50_norm", "dist_rolling_max_15m_norm", "dist_rolling_min_15m_norm",
        "dist_res_zone_top_15m_norm", "dist_res_zone_bottom_15m_norm", "dist_sup_zone_top_15m_norm", "dist_sup_zone_bottom_15m_norm",
        "dist_res_zone_top_30m_norm", "dist_res_zone_bottom_30m_norm", "dist_sup_zone_top_30m_norm", "dist_sup_zone_bottom_30m_norm",
        "dist_res_zone_top_4h_norm", "dist_res_zone_bottom_4h_norm", "dist_sup_zone_top_4h_norm", "dist_sup_zone_bottom_4h_norm",
        "dist_daily_eq_norm", "dist_pivot_norm", "dist_R1_norm", "dist_S1_norm"
    ]
    
    # Phase A: Oracle
    oracle = TemporalAttentionOracle(input_dim=len(feature_cols), seq_len=30).to(device)
    oracle.load_state_dict(torch.load("models/oracle/best_oracle.pth", map_location=device))
    oracle.eval()

    # Phase B: Manager
    manager = SAC.load("models/manager/saved/wfa_43/best_model.zip", device=device)
    feature_buffer = deque(maxlen=30)

    # 3. State Management Variables (The Missing Gates)
    bars_since_last_trade = 0
    trades_today = 0
    current_day = datetime.now(timezone.utc).date()
    
    # Static "Blind" Risk State for WhatsApp Signal Distribution
    simulated_equity_ratio = 1.0  
    simulated_drawdown_ratio = 0.0

    print("System Online. Awaiting market ticks...")

    # 4. Async Event Loop
    while True:
        # Pull latest tick from live MT5 feed
        timestamp, tick_row = feed.get_latest_tick()
        
        if timestamp is not None and tick_row is not None:
            # --- UTC Temporal Synchronization ---
            if timestamp.date() > current_day:
                current_day = timestamp.date()
                trades_today = 0
                print(f"[{timestamp}] 📅 New trading day detected (UTC). Daily execution limit reset to 0.")
                
            # Feed tick into the stateful feature engine
            latest_15m_features = feature_engine.process_m1_tick(timestamp, tick_row)
            
            # If a new 15m candle closed and features were calculated
            if latest_15m_features is not None:
                bars_since_last_trade += 1
                
                feature_vector = [latest_15m_features.get(c, 0.0) if not np.isnan(latest_15m_features.get(c, 0.0)) else 0.0 for c in feature_cols]
                feature_buffer.append(feature_vector)
                
                # Ensure buffer is full before inference
                if len(feature_buffer) == 30:
                    window_tensor = torch.FloatTensor(np.array(feature_buffer)).unsqueeze(0).to(device)
                    
                    # Phase A: Oracle Inference
                    with torch.no_grad():
                        logits = oracle(window_tensor)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        
                    prob_hold, prob_long, prob_short = probs[0], probs[1], probs[2]
                    
                    EXECUTION_THRESHOLD = 0.35
                    current_h4_trend = latest_15m_features.get("h4_trend", 0)
                    direction = 0

                    # Hybrid Execution: Relative Conviction + Macro Trend Gatekeeper
                    if prob_long > EXECUTION_THRESHOLD and prob_long > prob_short:
                        if current_h4_trend > 0:
                            direction = 1
                    elif prob_short > EXECUTION_THRESHOLD and prob_short > prob_long:
                        if current_h4_trend < 0:
                            direction = 2

                    # --- CRITICAL: TEMPORAL EXECUTION GATES ---
                    if direction != 0:
                        if bars_since_last_trade < 96: # 24-hour cooldown
                            print(f"[{timestamp}] ⏳ High-conviction signal detected, but blocked by 24h Cooldown.")
                            direction = 0  
                        elif trades_today >= 1:        # Absolute hard cap (1 trade/day)
                            print(f"[{timestamp}] 🛑 High-conviction signal detected, but blocked by Daily Limit.")
                            direction = 0  

                    if direction != 0:
                        # Construct 31-dim SAC Observation
                        obs = np.zeros(31, dtype=np.float32)
                        obs[:25] = feature_vector
                        obs[25] = prob_hold
                        obs[26] = prob_long
                        obs[27] = prob_short
                        
                        # Apply Blind State for independent signal sizing
                        obs[28] = simulated_equity_ratio 
                        obs[29] = simulated_drawdown_ratio
                        obs[30] = float(np.clip(bars_since_last_trade / 480.0, 0.0, 1.0))
                        
                        # Phase B: Manager Inference
                        action, _ = manager.predict(obs, deterministic=True)
                        size_val, tp_val, sl_val = action[1], action[2], action[3]
                        
                        # Calculate structural multipliers (1.0x - 3.0x Asymmetric Floor)
                        sl_mult = ((sl_val + 1.0) / 2.0) * 1.0 + 0.5
                        tp_mult = sl_mult * (((tp_val + 1.0) / 2.0) * 2.0 + 1.0)
                        
                        env_atr = latest_15m_features.get("env_atr", 1.0)
                        
                        signal_data = {
                            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "type": "Long" if direction == 1 else "Short",
                            "entry_price": round(tick_row["xau_close"], 3),
                            "sl_distance_pips": round((env_atr * sl_mult) * 10, 1),
                            "tp_distance_pips": round((env_atr * tp_mult) * 10, 1),
                            "confidence": round(prob_long if direction == 1 else prob_short, 3)
                        }
                        
                        print(f"[{timestamp}] 🚀 VALID SIGNAL DETECTED. Dispatching to WhatsApp Copilot...")
                        
                        # Fire-and-forget webhook task
                        asyncio.create_task(wa_manager.broadcast_signal(signal_data))
                        
                        bars_since_last_trade = 0
                        trades_today += 1 

        # Rest the loop to prevent CPU pinning. Polling MT5 4 times a second is optimal.
        await asyncio.sleep(0.25)


async def main():
    print("Initializing Meta Copilot Backend...")
    wa_manager = WhatsAppManager()
    
    # Launch the continuous background task to keep the Meta 24-Hour window open
    asyncio.create_task(wa_manager.check_and_send_reminders())
    
    # Launch the primary trading loop
    await live_trading_loop(wa_manager)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutting down live execution engine cleanly.")