import os
import time
import torch
import numpy as np
import pandas as pd
from datetime import timezone
from stable_baselines3 import SAC
from models.oracle.attention_net import TemporalAttentionOracle

class DualM1DataFeed:
    """Mimics a live MT5 terminal receiving ticks from multiple symbols simultaneously."""
    def __init__(self, xau_m1_path, dxy_m1_path):
        print("Initializing Synchronized Dual-Stream M1 Feed...")
        
        xau_df = self._parse_mt_csv(xau_m1_path).add_prefix("xau_")
        dxy_df = self._parse_mt_csv(dxy_m1_path).add_prefix("dxy_")
        
        self.master_stream = xau_df.join(dxy_df, how="inner")
        
        # --- THE FIX: PRE-SLICE THE DATA ---
        total_data = len(self.master_stream)
        oos_size = int(total_data * 0.20)  # We only want to test on the last 20%
        warmup_buffer = 3000               # 3000 ticks is plenty for the 800-period 15m EMA
        
        start_index = max(0, total_data - oos_size - warmup_buffer)
        self.master_stream = self.master_stream.iloc[start_index:]
        
        print(f"Dataset pre-sliced. Bypassing 1.5M in-sample ticks.")
        print(f"Total ticks to process (Warmup + OOS): {len(self.master_stream)}")

    def _parse_mt_csv(self, filepath):
        separator = "\t" if len(pd.read_csv(filepath, nrows=1, sep="\t").columns) > 1 else ","
        df = pd.read_csv(filepath, sep=separator)
        df.columns = [c.strip("<>").lower() for c in df.columns]
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
        df.set_index("datetime", inplace=True)
        return df[["open", "high", "low", "close"]] 

    def stream(self):
        """Yields one synchronized row of XAU and DXY data at a time."""
        for timestamp, row in self.master_stream.iterrows():
            yield timestamp, row

class StreamingFeatureEngine:
    """Stateful feature builder supporting full multi-timeframe zone logic."""
    def __init__(self, window_size=1000):
        self.history_limit = window_size 
        self.m1_buffer = []
        self.m15_history = pd.DataFrame()
        self.is_warmed_up = False
        
        # --- NEW GATEKEEPER ---
        self.last_closed_15m_mark = None 

    def process_m1_tick(self, timestamp, tick_row):
        self.m1_buffer.append(tick_row)
        
        # In live mode, this condition evaluates to True ~240 times a minute.
        if timestamp.minute % 15 == 0 and len(self.m1_buffer) > 0:
            
            # Standardize the timestamp to the exact 15-minute boundary
            # e.g., 15:00:23 becomes exactly 15:00:00
            current_15m_mark = timestamp.floor('15min')
            
            # THE FIX: Only close the candle IF we haven't already closed this exact block
            if self.last_closed_15m_mark != current_15m_mark:
                
                # Pass the clean boundary timestamp to prevent microsecond mismatches
                features = self._close_15m_candle(current_15m_mark)
                
                self.m1_buffer = [] 
                self.last_closed_15m_mark = current_15m_mark
                return features
                
        return None

    def _close_15m_candle(self, timestamp):
        m1_df = pd.DataFrame(self.m1_buffer)
        new_15m = pd.DataFrame({
            "open": [m1_df["xau_open"].iloc[0]],
            "high": [m1_df["xau_high"].max()],
            "low": [m1_df["xau_low"].min()],
            "close": [m1_df["xau_close"].iloc[-1]],
            "dxy_close": [m1_df["dxy_close"].iloc[-1]] 
        }, index=[timestamp])

        self.m15_history = pd.concat([self.m15_history, new_15m])
        
        if len(self.m15_history) > self.history_limit:
            self.m15_history = self.m15_history.iloc[-self.history_limit:]
            self.is_warmed_up = True

        if self.is_warmed_up:
            return self._calculate_current_features()
        return None

    def _calculate_current_features(self):
        df = self.m15_history.copy()
        
        # --- 1. Base Variables ---
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        df["env_atr"] = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1).rolling(14).mean()

        ema_50h = df["close"].ewm(span=200, adjust=False).mean()
        ema_200h = df["close"].ewm(span=800, adjust=False).mean()
        df["h4_trend"] = (ema_50h - ema_200h) / ema_200h

        h1_high = df["high"].rolling(window=4).max()
        h1_low = df["low"].rolling(window=4).min()
        df["h1_vol_regime"] = (h1_high - h1_low).rolling(window=96).rank(pct=True)

        weights = self._get_fractional_weights(0.45, 50)
        last_50_close = df["close"].iloc[-50:].values
        df.loc[df.index[-1], "close_frac_diff"] = np.dot(last_50_close, weights)[0]

        df["mom_1_norm"] = (df["close"] - df["close"].shift(1)) / df["env_atr"]
        df["mom_4_norm"] = (df["close"] - df["close"].shift(4)) / df["env_atr"]
        df["dxy_pct_change_15m"] = df["dxy_close"].pct_change()

        # --- 2. Multi-Timeframe Wick Zones & Levels ---
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["rolling_max_15m"] = df["high"].rolling(window=11, center=False).max()
        df["rolling_min_15m"] = df["low"].rolling(window=11, center=False).min()
        is_swing_high = df["high"].shift(5) == df["rolling_max_15m"]
        is_swing_low = df["low"].shift(5) == df["rolling_min_15m"]
        for col in ["res_zone_top_15m", "res_zone_bottom_15m", "sup_zone_top_15m", "sup_zone_bottom_15m"]:
            df[col] = np.nan
        df.loc[is_swing_high, "res_zone_top_15m"] = df["high"].shift(5)
        df.loc[is_swing_high, "res_zone_bottom_15m"] = df[["open", "close"]].shift(5).max(axis=1)
        df.loc[is_swing_low, "sup_zone_bottom_15m"] = df["low"].shift(5)
        df.loc[is_swing_low, "sup_zone_top_15m"] = df[["open", "close"]].shift(5).min(axis=1)
        for col in ["res_zone_top_15m", "res_zone_bottom_15m", "sup_zone_top_15m", "sup_zone_bottom_15m"]:
            df[col] = df[col].ffill()

        df_30m = df.resample("30min").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        df_30m["rolling_max"] = df_30m["high"].rolling(window=11, center=False).max()
        df_30m["rolling_min"] = df_30m["low"].rolling(window=11, center=False).min()
        sh_30 = df_30m["high"].shift(5) == df_30m["rolling_max"]
        sl_30 = df_30m["low"].shift(5) == df_30m["rolling_min"]
        for col in ["res_zone_top_30m", "res_zone_bottom_30m", "sup_zone_top_30m", "sup_zone_bottom_30m"]:
            df_30m[col] = np.nan
        df_30m.loc[sh_30, "res_zone_top_30m"] = df_30m["high"].shift(5)
        df_30m.loc[sh_30, "res_zone_bottom_30m"] = df_30m[["open", "close"]].shift(5).max(axis=1)
        df_30m.loc[sl_30, "sup_zone_bottom_30m"] = df_30m["low"].shift(5)
        df_30m.loc[sl_30, "sup_zone_top_30m"] = df_30m[["open", "close"]].shift(5).min(axis=1)
        for col in ["res_zone_top_30m", "res_zone_bottom_30m", "sup_zone_top_30m", "sup_zone_bottom_30m"]:
            df_30m[col] = df_30m[col].ffill()

        df_4h = df.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        df_4h["rolling_max"] = df_4h["high"].rolling(window=11, center=False).max()
        df_4h["rolling_min"] = df_4h["low"].rolling(window=11, center=False).min()
        sh_4h = df_4h["high"].shift(5) == df_4h["rolling_max"]
        sl_4h = df_4h["low"].shift(5) == df_4h["rolling_min"]
        for col in ["res_zone_top_4h", "res_zone_bottom_4h", "sup_zone_top_4h", "sup_zone_bottom_4h"]:
            df_4h[col] = np.nan
        df_4h.loc[sh_4h, "res_zone_top_4h"] = df_4h["high"].shift(5)
        df_4h.loc[sh_4h, "res_zone_bottom_4h"] = df_4h[["open", "close"]].shift(5).max(axis=1)
        df_4h.loc[sl_4h, "sup_zone_bottom_4h"] = df_4h["low"].shift(5)
        df_4h.loc[sl_4h, "sup_zone_top_4h"] = df_4h[["open", "close"]].shift(5).min(axis=1)
        for col in ["res_zone_top_4h", "res_zone_bottom_4h", "sup_zone_top_4h", "sup_zone_bottom_4h"]:
            df_4h[col] = df_4h[col].ffill()

        daily = df.resample("D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        daily["prev_high"] = daily["high"].shift(1)
        daily["prev_low"] = daily["low"].shift(1)
        daily["prev_close"] = daily["close"].shift(1)
        daily["daily_eq"] = (daily["prev_high"] + daily["prev_low"]) / 2.0
        daily["pivot"] = (daily["prev_high"] + daily["prev_low"] + daily["prev_close"]) / 3.0
        daily["R1"] = (2 * daily["pivot"]) - daily["prev_low"]
        daily["S1"] = (2 * daily["pivot"]) - daily["prev_high"]

        df = pd.merge_asof(df, df_30m[["res_zone_top_30m", "res_zone_bottom_30m", "sup_zone_top_30m", "sup_zone_bottom_30m"]], left_index=True, right_index=True)
        df = pd.merge_asof(df, df_4h[["res_zone_top_4h", "res_zone_bottom_4h", "sup_zone_top_4h", "sup_zone_bottom_4h"]], left_index=True, right_index=True)
        df = pd.merge_asof(df, daily[["daily_eq", "pivot", "R1", "S1"]], left_index=True, right_index=True)

        # --- 3. Distance Normalization ---
        price_level_cols = [
            "ema_50", "rolling_max_15m", "rolling_min_15m",
            "res_zone_top_15m", "res_zone_bottom_15m", "sup_zone_top_15m", "sup_zone_bottom_15m",
            "res_zone_top_30m", "res_zone_bottom_30m", "sup_zone_top_30m", "sup_zone_bottom_30m",
            "res_zone_top_4h", "res_zone_bottom_4h", "sup_zone_top_4h", "sup_zone_bottom_4h",
            "daily_eq", "pivot", "R1", "S1"
        ]
        for col in price_level_cols:
            df[f"dist_{col}_norm"] = (df[col] - df["close"]) / df["env_atr"]

        return df.iloc[-1].to_dict()

    def _get_fractional_weights(self, d: float, size: int) -> np.ndarray:
        w = [1.0]
        for k in range(1, size):
            w_ = -w[-1] / k * (d - k + 1)
            w.append(w_)
        return np.array(w[::-1]).reshape(-1, 1)

class M1HighFidelitySimulator:
    def __init__(self, xau_path, dxy_path, oracle_path, manager_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Initializing Ultra-Fidelity M1 Prop Firm Simulator...")

        self.feed = DualM1DataFeed(xau_path, dxy_path)
        self.engine = StreamingFeatureEngine(window_size=1000)

        self.commission_per_lot = 5.00
        self.spread_pips = 2.0
        self.pip_value_per_lot = 10.0

        self.initial_balance = 5000.0
        self.fixed_risk_usd = 20.00
        self.consistency_cap_usd = 37.50
        self.guardian_shield_loss = -50.00
        
        self.max_daily_loss = 150.00
        self.max_trailing_loss = 250.00
        self.profit_target = 5250.00
        
        # New trade pacing limits
        self.max_trades_per_day = 5
        self.min_bars_between_trades = 4

        self._load_models(oracle_path, manager_path)

    def _load_models(self, oracle_path, manager_path):
        self.feature_cols = [
            "h4_trend", "h1_vol_regime", "close_frac_diff", "mom_1_norm", "mom_4_norm", "dxy_pct_change_15m",
            "dist_ema_50_norm", "dist_rolling_max_15m_norm", "dist_rolling_min_15m_norm",
            "dist_res_zone_top_15m_norm", "dist_res_zone_bottom_15m_norm", "dist_sup_zone_top_15m_norm", "dist_sup_zone_bottom_15m_norm",
            "dist_res_zone_top_30m_norm", "dist_res_zone_bottom_30m_norm", "dist_sup_zone_top_30m_norm", "dist_sup_zone_bottom_30m_norm",
            "dist_res_zone_top_4h_norm", "dist_res_zone_bottom_4h_norm", "dist_sup_zone_top_4h_norm", "dist_sup_zone_bottom_4h_norm",
            "dist_daily_eq_norm", "dist_pivot_norm", "dist_R1_norm", "dist_S1_norm"
        ]

        self.oracle = TemporalAttentionOracle(
            input_dim=len(self.feature_cols), seq_len=30
        ).to(self.device)
        self.oracle.load_state_dict(torch.load(oracle_path, map_location=self.device))
        self.oracle.eval()

        self.manager = SAC.load(manager_path, device=self.device)
        
        from collections import deque
        self.feature_buffer = deque(maxlen=30)

    def is_restricted_time(self, current_time: pd.Timestamp) -> bool:
        if (current_time.hour == 23 and current_time.minute >= 45) or (
            current_time.hour == 0 and current_time.minute <= 30
        ):
            return True
        return False

    def run_simulation(self):
        total_ticks = len(self.feed.master_stream)
        
        # Since Data is pre-sliced, we use the warmup buffer index
        holdout_start_idx = 3000 
        
        print(f"Executing Sequential Challenge Yield Tester...")
        print(f"Enforcing OOS Firewall at Index: {holdout_start_idx}")

        # Core State
        equity = self.initial_balance
        high_water_mark = self.initial_balance
        daily_start_equity = self.initial_balance
        current_day = None
        
        trading_locked_for_day = False
        trades_today = 0
        
        # Challenge Yield Trackers
        challenge_history = []
        current_challenge_start_time = None
        trades_in_current_challenge = 0
        
        journal = []
        active_trade = None
        pending_signal = None
        bars_since_last_trade = 0 
        latency_logs = []

        # START EVENT LOOP
        for idx, (timestamp, tick_row) in enumerate(self.feed.stream()):
            if idx % 10000 == 0:
              progress_pct = (idx / total_ticks) * 100
              status = "WARMUP (In-Sample)" if idx < holdout_start_idx else "ACTIVE (Out-of-Sample)"
              print(f"[{timestamp}] Progress: {progress_pct:.2f}% ({idx}/{total_ticks}) | Status: {status} | Equity: ${equity:.2f}")
            
            # --- 1. STATEFUL FEATURE FEED (Warm-Up Mode) ---
            latest_15m_features = self.engine.process_m1_tick(timestamp, tick_row)
            if latest_15m_features is not None:
                feature_vector = [latest_15m_features.get(c, 0.0) if not np.isnan(latest_15m_features.get(c, 0.0)) else 0.0 for c in self.feature_cols]
                self.feature_buffer.append(feature_vector)
            
            # --- OOS FIREWALL GATE ---
            if idx < holdout_start_idx:
                continue
                
            if current_challenge_start_time is None:
                current_challenge_start_time = timestamp
            
            # --- 2. UTC Temporal Synchronization ---
            if current_day is None:
                current_day = timestamp.date()

            if timestamp.date() > current_day:
                daily_start_equity = equity
                current_day = timestamp.date()
                trading_locked_for_day = False
                trades_today = 0 # Daily counter reset

            # --- 3. M1 TICK-LEVEL TRADE MANAGEMENT ---
            account_failed = False
            account_passed = False
            
            if active_trade is not None:
                trade_closed = False
                exit_price = 0.0
                exit_reason = ""

                current_price = tick_row["xau_close"]
                current_distance_pips = (current_price - active_trade["entry"]) * 10.0
                if active_trade["type"] == "Short":
                    current_distance_pips = -current_distance_pips
                
                floating_pnl = current_distance_pips * self.pip_value_per_lot * active_trade["lot_size"] - active_trade["total_friction"]
                current_floating_equity = equity + floating_pnl

                if current_floating_equity <= (daily_start_equity - self.max_daily_loss):
                    trade_closed, exit_price, exit_reason = (True, current_price, "Daily Drawdown Breached")
                    trading_locked_for_day = True
                elif current_floating_equity <= (high_water_mark - self.max_trailing_loss):
                    trade_closed, exit_price, exit_reason = (True, current_price, "Trailing Drawdown Breached")
                    account_failed = True
                elif current_floating_equity >= self.profit_target:
                    trade_closed, exit_price, exit_reason = (True, current_price, "Profit Target Reached")
                    account_passed = True
                elif floating_pnl <= self.guardian_shield_loss:
                    trade_closed, exit_price, exit_reason = (True, current_price, "Guardian Shield (1% Loss)")
                elif floating_pnl >= self.consistency_cap_usd:
                    trade_closed, exit_price, exit_reason = (True, current_price, "Consistency Hard Clip (15% Rule)")
                elif active_trade["type"] == "Long":
                    if tick_row["xau_low"] <= active_trade["sl"]:
                        trade_closed, exit_price, exit_reason = (True, active_trade["sl"], "Stop Loss")
                    elif tick_row["xau_high"] >= active_trade["tp"]:
                        trade_closed, exit_price, exit_reason = (True, active_trade["tp"], "Take Profit")
                elif active_trade["type"] == "Short":
                    if tick_row["xau_high"] >= active_trade["sl"]:
                        trade_closed, exit_price, exit_reason = (True, active_trade["sl"], "Stop Loss")
                    elif tick_row["xau_low"] <= active_trade["tp"]:
                        trade_closed, exit_price, exit_reason = (True, active_trade["tp"], "Take Profit")

                if trade_closed:
                    pip_diff = (exit_price - active_trade["entry"]) * 10
                    if active_trade["type"] == "Short":
                        pip_diff *= -1

                    gross_pnl = (pip_diff * self.pip_value_per_lot * active_trade["lot_size"])
                    net_pnl = gross_pnl - active_trade["total_friction"]
                    equity += net_pnl
                    trades_in_current_challenge += 1

                    if equity > high_water_mark:
                        high_water_mark = equity

                    journal.append({
                        "Entry_Time": active_trade["time"],
                        "Exit_Time": timestamp,
                        "Type": active_trade["type"],
                        "Entry_Price": round(active_trade["entry"], 3),
                        "Exit_Price": round(exit_price, 3),
                        "Net_PnL": round(net_pnl, 2),
                        "Reason": exit_reason,
                    })
                    active_trade = None
                    continue

            # Check Global Breakers for Idle state
            if active_trade is None:
                if equity <= (daily_start_equity - self.max_daily_loss):
                    trading_locked_for_day = True
                if equity <= (high_water_mark - self.max_trailing_loss):
                    account_failed = True
                if equity >= self.profit_target:
                    account_passed = True

            # ==========================================
            # SEQUENTIAL RESTART LOGIC
            # ==========================================
            if account_failed or account_passed:
                result = "PASSED" if account_passed else "FAILED"
                print(f"[{timestamp}] Challenge {result}! Final Equity: ${equity:.2f} | Trades: {trades_in_current_challenge}")
                
                challenge_history.append({
                    "Start_Time": current_challenge_start_time,
                    "End_Time": timestamp,
                    "Result": result,
                    "Final_Equity": round(equity, 2),
                    "Trades_Taken": trades_in_current_challenge
                })
                
                # Instantly reset all tracking variables to buy a new challenge
                equity = self.initial_balance
                high_water_mark = self.initial_balance
                daily_start_equity = self.initial_balance
                trading_locked_for_day = False
                current_challenge_start_time = timestamp
                trades_in_current_challenge = 0
                trades_today = 0 # Daily counter reset
                active_trade = None
                pending_signal = None
                continue # Skip the rest of this tick to start fresh on the next

            # --- 4. EXECUTE PENDING QUEUE ---
            if pending_signal is not None and active_trade is None:
                fill_price = tick_row["xau_open"]
                sl_distance_pips = max(pending_signal["sl_distance"], 10.0)
                
                theoretical_lot_size = self.fixed_risk_usd / (sl_distance_pips * self.pip_value_per_lot)
                lot_size = np.clip(round(theoretical_lot_size, 2), 0.01, 100.0)
                
                total_friction = (lot_size * self.commission_per_lot) + (lot_size * self.spread_pips * self.pip_value_per_lot)
                
                active_trade = {
                    "time": timestamp,
                    "type": pending_signal["type"],
                    "entry": fill_price,
                    "sl": (fill_price - (sl_distance_pips * 0.1) if pending_signal["type"] == "Long" else fill_price + (sl_distance_pips * 0.1)),
                    "tp": (fill_price + (pending_signal["tp_distance"] * 0.1) if pending_signal["type"] == "Long" else fill_price - (pending_signal["tp_distance"] * 0.1)),
                    "lot_size": lot_size,
                    "total_friction": total_friction,
                }
                bars_since_last_trade = 0
                trades_today += 1 # Execution recorded for daily limits
                pending_signal = None
                continue

            # --- 5. NEURAL INFERENCE (Triggered Only on 15m Close) ---
            if latest_15m_features is not None:
                bars_since_last_trade += 1
                
                if trading_locked_for_day or self.is_restricted_time(timestamp):
                    continue

                if len(self.feature_buffer) == 30:
                    start_time = time.perf_counter()

                    window_tensor = torch.FloatTensor(np.array(self.feature_buffer)).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        logits = self.oracle(window_tensor)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    prob_hold, prob_long, prob_short = probs[0], probs[1], probs[2]

                    EXECUTION_THRESHOLD = 0.35
                    current_h4_trend = latest_15m_features.get("h4_trend", 0)
                    direction = 0

                    if prob_long > EXECUTION_THRESHOLD and prob_long > prob_short:
                        if current_h4_trend > 0:
                            direction = 1
                    elif prob_short > EXECUTION_THRESHOLD and prob_short > prob_long:
                        if current_h4_trend < 0:
                            direction = 2

                    # --- THE PROPER GATING FIX ---
                    if direction != 0:
                        if bars_since_last_trade < self.min_bars_between_trades:
                            direction = 0  # Blocked by 1-hour cooldown
                        elif trades_today >= self.max_trades_per_day:
                            direction = 0  # Blocked by 5-trade daily limit

                    if direction != 0:
                        obs = np.zeros(31, dtype=np.float32)
                        
                        obs[:25] = feature_vector
                        obs[25] = prob_hold
                        obs[26] = prob_long
                        obs[27] = prob_short
                        obs[28] = float(np.clip(equity / self.initial_balance, 0.0, 10.0))
                        obs[29] = float(np.clip((high_water_mark - equity) / high_water_mark, 0.0, 1.0))
                        obs[30] = float(np.clip(bars_since_last_trade / 480.0, 0.0, 1.0))
                        
                        action, _ = self.manager.predict(obs, deterministic=True)
                        size_val, tp_val, sl_val = action[1], action[2], action[3]
                        
                        sl_mult = ((sl_val + 1.0) / 2.0) * 1.0 + 0.5
                        tp_mult = sl_mult * (((tp_val + 1.0) / 2.0) * 2.0 + 1.0)
                        
                        pending_signal = {
                            "type": "Long" if direction == 1 else "Short",
                            "sl_distance": (latest_15m_features.get("env_atr", 1.0) * sl_mult) * 10,
                            "tp_distance": (latest_15m_features.get("env_atr", 1.0) * tp_mult) * 10
                        }

                    end_time = time.perf_counter()
                    latency_ms = (end_time - start_time) * 1000
                    latency_logs.append(latency_ms)

        # Print Final Report
        yield_df = pd.DataFrame(challenge_history)
        print("\n" + "=" * 60)
        print(" SEQUENTIAL CHALLENGE YIELD REPORT (OOS Period) ")
        print("=" * 60)
        
        if not yield_df.empty:
            total_attempts = len(yield_df)
            passes = len(yield_df[yield_df['Result'] == 'PASSED'])
            fails = len(yield_df[yield_df['Result'] == 'FAILED'])
            
            print(f"Total Challenges Attempted: {total_attempts}")
            print(f"Challenges PASSED:          {passes}")
            print(f"Challenges FAILED:          {fails}")
            print(f"Yield Winrate:              {(passes/total_attempts)*100:.2f}%")
            print("-" * 60)
            print("Detailed Run History:")
            print(yield_df.to_string(index=False))
        else:
            print("No challenges completed. The OOS period may be too short, or no trades were taken.")
            
        print("=" * 60)
        
        return yield_df

if __name__ == "__main__":
    XAU = "data/raw/XAUUSDr_M1.csv"
    DXY = "data/raw/USDIndex_M1.csv"
    ORACLE = "models/oracle/best_oracle.pth"
    MANAGER = "models/manager/saved/wfa_43/best_model.zip"
    
    sim = M1HighFidelitySimulator(XAU, DXY, ORACLE, MANAGER)
    sim.run_simulation()