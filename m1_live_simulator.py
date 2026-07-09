import os
import time
import numpy as np
import pandas as pd
from datetime import timezone
import onnxruntime as ort

class DualM1DataFeed:
    """Mimics a live MT5 terminal receiving ticks from multiple symbols simultaneously."""
    def __init__(self, xau_m1_path, dxy_m1_path):
        print("Initializing Synchronized Dual-Stream M1 Feed...")
        
        xau_df = self._parse_mt_csv(xau_m1_path).add_prefix("xau_")
        dxy_df = self._parse_mt_csv(dxy_m1_path).add_prefix("dxy_")
        
        self.master_stream = xau_df.join(dxy_df, how="inner")
        
        total_data = len(self.master_stream)
        oos_size = int(total_data * 0.20)  
        warmup_buffer = 3000               
        
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
        for timestamp, row in self.master_stream.iterrows():
            yield timestamp, row


class StreamingFeatureEngine:
    """Stateful feature builder aligned exactly with build_features.py"""
    def __init__(self, window_size=1000):
        self.history_limit = window_size 
        self.m1_buffer = []
        self.m15_history = pd.DataFrame()
        self.is_warmed_up = False
        self.last_closed_15m_mark = None 

    def process_m1_tick(self, timestamp, tick_row):
        self.m1_buffer.append(tick_row)
        
        if timestamp.minute % 15 == 0 and len(self.m1_buffer) > 0:
            current_15m_mark = timestamp.floor('15min')
            if self.last_closed_15m_mark != current_15m_mark:
                features = self._close_15m_candle(current_15m_mark)
                self.m1_buffer = [] 
                self.last_closed_15m_mark = current_15m_mark
                return features
        return None

    def _close_15m_candle(self, timestamp):
        m1_df = pd.DataFrame(self.m1_buffer)
        
        # PERFECT ALIGNMENT: Using exact column names generated during training
        new_15m = pd.DataFrame({
            "xau_open": [m1_df["xau_open"].iloc[0]],
            "xau_high": [m1_df["xau_high"].max()],
            "xau_low": [m1_df["xau_low"].min()],
            "xau_close": [m1_df["xau_close"].iloc[-1]],
            "dxy_close": [m1_df["dxy_close"].iloc[-1]] 
        }, index=[timestamp])

        self.m15_history = pd.concat([self.m15_history, new_15m])
        
        if len(self.m15_history) > self.history_limit:
            self.m15_history = self.m15_history.iloc[-self.history_limit:]
            
        if len(self.m15_history) >= 200:
            self.is_warmed_up = True

        if self.is_warmed_up:
            return self._calculate_current_features()
        return None

    def _calculate_current_features(self):
        df = self.m15_history.copy()
        
        high_low = df["xau_high"] - df["xau_low"]
        high_close = np.abs(df["xau_high"] - df["xau_close"].shift())
        low_close = np.abs(df["xau_low"] - df["xau_close"].shift())
        df["env_atr"] = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1).rolling(14).mean()

        df["h4_ema"] = df["xau_close"].ewm(span=800, adjust=False).mean()
        df["h4_trend"] = np.where(df["xau_close"] > df["h4_ema"], 1.0, -1.0)
        
        df["close_frac_diff"] = np.log(df["xau_close"] / df["xau_close"].shift(1))
        df["dxy_pct_change_15m"] = df["dxy_close"].pct_change()
        
        df['mom_1'] = df['xau_close'].diff(1)
        df['mom_4'] = df['xau_close'].diff(4)
        df['mom_1_norm'] = (df['mom_1'] - df['mom_1'].rolling(1000).mean()) / df['mom_1'].rolling(1000).std()
        df['mom_4_norm'] = (df['mom_4'] - df['mom_4'].rolling(1000).mean()) / df['mom_4'].rolling(1000).std()

        df["h1_vol_regime"] = df["env_atr"] / df["env_atr"].rolling(64).mean()

        df['ema_50'] = df['xau_close'].ewm(span=50, adjust=False).mean()
        df['dist_ema_50'] = (df['xau_close'] - df['ema_50']) / df['xau_close']
        df['dist_ema_50_norm'] = (df['dist_ema_50'] - df['dist_ema_50'].rolling(1000).mean()) / df['dist_ema_50'].rolling(1000).std()

        df['rolling_max_15m'] = df['xau_high'].rolling(14).max()
        df['rolling_min_15m'] = df['xau_low'].rolling(14).min()
        df['dist_rolling_max_15m_norm'] = (df['rolling_max_15m'] - df['xau_close']) / df['env_atr']
        df['dist_rolling_min_15m_norm'] = (df['xau_close'] - df['rolling_min_15m']) / df['env_atr']

        # Multi-Timeframe Wick Zones
        for col in ["res_zone_top_15m", "res_zone_bottom_15m", "sup_zone_top_15m", "sup_zone_bottom_15m"]:
            df[col] = np.nan
        is_swing_high = df["xau_high"].shift(5) == df["rolling_max_15m"]
        is_swing_low = df["xau_low"].shift(5) == df["rolling_min_15m"]
        df.loc[is_swing_high, "res_zone_top_15m"] = df["xau_high"].shift(5)
        df.loc[is_swing_high, "res_zone_bottom_15m"] = df[["xau_open", "xau_close"]].shift(5).max(axis=1)
        df.loc[is_swing_low, "sup_zone_bottom_15m"] = df["xau_low"].shift(5)
        df.loc[is_swing_low, "sup_zone_top_15m"] = df[["xau_open", "xau_close"]].shift(5).min(axis=1)
        for col in ["res_zone_top_15m", "res_zone_bottom_15m", "sup_zone_top_15m", "sup_zone_bottom_15m"]:
            df[col] = df[col].ffill()

        df_30m = df.resample("30min").agg({"xau_open": "first", "xau_high": "max", "xau_low": "min", "xau_close": "last"}).dropna()
        df_30m["rolling_max"] = df_30m["xau_high"].rolling(window=11, center=False).max()
        df_30m["rolling_min"] = df_30m["xau_low"].rolling(window=11, center=False).min()
        sh_30 = df_30m["xau_high"].shift(5) == df_30m["rolling_max"]
        sl_30 = df_30m["xau_low"].shift(5) == df_30m["rolling_min"]
        for col in ["res_zone_top_30m", "res_zone_bottom_30m", "sup_zone_top_30m", "sup_zone_bottom_30m"]:
            df_30m[col] = np.nan
        df_30m.loc[sh_30, "res_zone_top_30m"] = df_30m["xau_high"].shift(5)
        df_30m.loc[sh_30, "res_zone_bottom_30m"] = df_30m[["xau_open", "xau_close"]].shift(5).max(axis=1)
        df_30m.loc[sl_30, "sup_zone_bottom_30m"] = df_30m["xau_low"].shift(5)
        df_30m.loc[sl_30, "sup_zone_top_30m"] = df_30m[["xau_open", "xau_close"]].shift(5).min(axis=1)
        for col in ["res_zone_top_30m", "res_zone_bottom_30m", "sup_zone_top_30m", "sup_zone_bottom_30m"]:
            df_30m[col] = df_30m[col].ffill()

        df_4h = df.resample("4h").agg({"xau_open": "first", "xau_high": "max", "xau_low": "min", "xau_close": "last"}).dropna()
        df_4h["rolling_max"] = df_4h["xau_high"].rolling(window=11, center=False).max()
        df_4h["rolling_min"] = df_4h["xau_low"].rolling(window=11, center=False).min()
        sh_4h = df_4h["xau_high"].shift(5) == df_4h["rolling_max"]
        sl_4h = df_4h["xau_low"].shift(5) == df_4h["rolling_min"]
        for col in ["res_zone_top_4h", "res_zone_bottom_4h", "sup_zone_top_4h", "sup_zone_bottom_4h"]:
            df_4h[col] = np.nan
        df_4h.loc[sh_4h, "res_zone_top_4h"] = df_4h["xau_high"].shift(5)
        df_4h.loc[sh_4h, "res_zone_bottom_4h"] = df_4h[["xau_open", "xau_close"]].shift(5).max(axis=1)
        df_4h.loc[sl_4h, "sup_zone_bottom_4h"] = df_4h["xau_low"].shift(5)
        df_4h.loc[sl_4h, "sup_zone_top_4h"] = df_4h[["xau_open", "xau_close"]].shift(5).min(axis=1)
        for col in ["res_zone_top_4h", "res_zone_bottom_4h", "sup_zone_top_4h", "sup_zone_bottom_4h"]:
            df_4h[col] = df_4h[col].ffill()

        daily = df.resample("D").agg({"xau_open": "first", "xau_high": "max", "xau_low": "min", "xau_close": "last"}).dropna()
        daily["prev_high"] = daily["xau_high"].shift(1)
        daily["prev_low"] = daily["xau_low"].shift(1)
        daily["prev_close"] = daily["xau_close"].shift(1)
        daily["daily_eq"] = (daily["prev_high"] + daily["prev_low"]) / 2.0
        daily["pivot"] = (daily["prev_high"] + daily["prev_low"] + daily["prev_close"]) / 3.0
        daily["R1"] = (2 * daily["pivot"]) - daily["prev_low"]
        daily["S1"] = (2 * daily["pivot"]) - daily["prev_high"]

        df = pd.merge_asof(df, df_30m[["res_zone_top_30m", "res_zone_bottom_30m", "sup_zone_top_30m", "sup_zone_bottom_30m"]], left_index=True, right_index=True)
        df = pd.merge_asof(df, df_4h[["res_zone_top_4h", "res_zone_bottom_4h", "sup_zone_top_4h", "sup_zone_bottom_4h"]], left_index=True, right_index=True)
        df = pd.merge_asof(df, daily[["daily_eq", "pivot", "R1", "S1"]], left_index=True, right_index=True)

        price_level_cols = [
            "res_zone_top_15m", "res_zone_bottom_15m", "sup_zone_top_15m", "sup_zone_bottom_15m",
            "res_zone_top_30m", "res_zone_bottom_30m", "sup_zone_top_30m", "sup_zone_bottom_30m",
            "res_zone_top_4h", "res_zone_bottom_4h", "sup_zone_top_4h", "sup_zone_bottom_4h",
            "daily_eq", "pivot", "R1", "S1"
        ]
        for col in price_level_cols:
            df[f"dist_{col}_norm"] = (df[col] - df["xau_close"]) / df["env_atr"]

        # Default Probability States (Updated dynamically by the Oracle)
        df['prob_long'] = 0.0
        df['prob_short'] = 0.0
        df['prob_hold'] = 1.0

        return df.iloc[-1].to_dict()


class M1HighFidelitySimulator:
    def __init__(self, xau_path, dxy_path, oracle_onnx_path, manager_onnx_path):
        print("Initializing Ultra-Fidelity M1 Prop Firm Simulator (ONNX NATIVE)...")

        self.feed = DualM1DataFeed(xau_path, dxy_path)
        self.engine = StreamingFeatureEngine(window_size=1000)

        self.commission_per_lot = 5.00
        self.spread_pips = 2.0
        self.pip_value_per_lot = 10.0

        self.initial_balance = 5000.0
        self.fixed_risk_usd = 20.00
        self.guardian_shield_loss = -50.00
        
        self.max_daily_loss = 150.00
        self.max_trailing_loss = 250.00
        self.profit_target = 5250.00
        
        self.max_trades_per_day = 1

        self._load_onnx_models(oracle_onnx_path, manager_onnx_path)

    def _load_onnx_models(self, oracle_onnx_path, manager_onnx_path):
        # 39 EXACT features aligned with the diagnostic output
        self.feature_cols = [
            "xau_open", "xau_high", "xau_low", "xau_close", 
            "h4_ema", "h4_trend", "close_frac_diff", "dxy_pct_change_15m", 
            "mom_1", "mom_4", "mom_1_norm", "mom_4_norm", 
            "h1_vol_regime", "ema_50", "dist_ema_50", "dist_ema_50_norm", 
            "rolling_max_15m", "rolling_min_15m", "dist_rolling_max_15m_norm", "dist_rolling_min_15m_norm", 
            "dist_res_zone_top_15m_norm", "dist_res_zone_bottom_15m_norm", "dist_sup_zone_top_15m_norm", "dist_sup_zone_bottom_15m_norm", 
            "dist_res_zone_top_30m_norm", "dist_res_zone_bottom_30m_norm", "dist_sup_zone_top_30m_norm", "dist_sup_zone_bottom_30m_norm", 
            "dist_res_zone_top_4h_norm", "dist_res_zone_bottom_4h_norm", "dist_sup_zone_top_4h_norm", "dist_sup_zone_bottom_4h_norm", 
            "dist_daily_eq_norm", "dist_pivot_norm", "dist_R1_norm", "dist_S1_norm", 
            "prob_long", "prob_short", "prob_hold"
        ]

        # Initialize lightweight ONNX Runtime Sessions (CPU optimized)
        self.oracle_session = ort.InferenceSession(oracle_onnx_path, providers=['CPUExecutionProvider'])
        self.manager_session = ort.InferenceSession(manager_onnx_path, providers=['CPUExecutionProvider'])
        
        from collections import deque
        self.feature_buffer = deque(maxlen=30)

    def run_simulation(self):
        total_ticks = len(self.feed.master_stream)
        holdout_start_idx = 3000 
        
        print(f"Executing Sequential Challenge Yield Tester...")
        print(f"Enforcing OOS Firewall at Index: {holdout_start_idx}")

        equity = self.initial_balance
        high_water_mark = self.initial_balance
        daily_start_equity = self.initial_balance
        current_day = None
        
        trading_locked_for_day = False
        trades_today = 0
        
        challenge_history = []
        current_challenge_start_time = None
        trades_in_current_challenge = 0
        
        journal = []
        active_trade = None
        pending_signal = None
        bars_since_last_trade = 0 

        for idx, (timestamp, tick_row) in enumerate(self.feed.stream()):
            if idx % 10000 == 0:
              progress_pct = (idx / total_ticks) * 100
              status = "WARMUP (In-Sample)" if idx < holdout_start_idx else "ACTIVE (Out-of-Sample)"
              print(f"[{timestamp}] Progress: {progress_pct:.2f}% ({idx}/{total_ticks}) | Status: {status} | Equity: ${equity:.2f}")
            
            latest_15m_features = self.engine.process_m1_tick(timestamp, tick_row)
            if latest_15m_features is not None:
                feature_vector = [latest_15m_features.get(c, 0.0) if not np.isnan(latest_15m_features.get(c, 0.0)) else 0.0 for c in self.feature_cols]
                self.feature_buffer.append(feature_vector)
            
            if idx < holdout_start_idx:
                continue
                
            if current_challenge_start_time is None:
                current_challenge_start_time = timestamp
            
            if current_day is None:
                current_day = timestamp.date()

            if timestamp.date() > current_day:
                daily_start_equity = equity
                current_day = timestamp.date()
                trading_locked_for_day = False
                trades_today = 0 

            account_failed = False
            account_passed = False
            
            if active_trade is not None:
                trade_closed = False
                exit_price = 0.0
                exit_reason = ""

                if active_trade["type"] == "Long":
                    worst_price = tick_row["xau_low"] 
                    worst_distance_pips = (worst_price - active_trade["entry"]) * 10.0
                    best_price = tick_row["xau_high"] 
                else: 
                    worst_price = tick_row["xau_high"] + (self.spread_pips * 0.1) 
                    worst_distance_pips = (active_trade["entry"] - worst_price) * 10.0
                    best_price = tick_row["xau_low"] 

                worst_floating_pnl = (worst_distance_pips * self.pip_value_per_lot * active_trade["lot_size"]) - active_trade["total_friction"]
                worst_floating_equity = equity + worst_floating_pnl

                if worst_floating_equity <= (daily_start_equity - self.max_daily_loss):
                    trade_closed, exit_price, exit_reason = (True, worst_price, "Daily Drawdown Breached (Intra-minute)")
                    trading_locked_for_day = True
                elif worst_floating_equity <= (high_water_mark - self.max_trailing_loss):
                    trade_closed, exit_price, exit_reason = (True, worst_price, "Trailing Drawdown Breached (Intra-minute)")
                    account_failed = True
                elif worst_floating_pnl <= self.guardian_shield_loss:
                    trade_closed, exit_price, exit_reason = (True, worst_price, "Guardian Shield (1% Loss)")
                
                elif active_trade["type"] == "Long":
                    if worst_price <= active_trade["sl"]:
                        trade_closed, exit_price, exit_reason = (True, active_trade["sl"], "Stop Loss")
                    elif best_price >= active_trade["tp"]:
                        trade_closed, exit_price, exit_reason = (True, active_trade["tp"], "Take Profit")
                elif active_trade["type"] == "Short":
                    if worst_price >= active_trade["sl"]:
                        trade_closed, exit_price, exit_reason = (True, active_trade["sl"], "Stop Loss")
                    elif best_price <= active_trade["tp"]:
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
                        "Net_PnL": round(net_pnl, 2),
                        "Equity": round(equity, 2),
                        "Reason": exit_reason,
                    })
                    active_trade = None
                    continue

            if active_trade is None:
                if equity <= (daily_start_equity - self.max_daily_loss):
                    trading_locked_for_day = True
                if equity <= (high_water_mark - self.max_trailing_loss):
                    account_failed = True
                if equity >= self.profit_target:
                    account_passed = True

            if account_failed or account_passed:
                result = "PASSED" if account_passed else "FAILED"
                print(f"[{timestamp}] Challenge {result}! Final Equity: ${equity:.2f} | Trades: {trades_in_current_challenge}")
                
                challenge_history.append({
                    "Result": result,
                    "Final_Equity": round(equity, 2),
                })
                
                equity = self.initial_balance
                high_water_mark = self.initial_balance
                daily_start_equity = self.initial_balance
                trading_locked_for_day = False
                current_challenge_start_time = timestamp
                trades_in_current_challenge = 0
                trades_today = 0 
                active_trade = None
                pending_signal = None
                continue 

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
                trades_today += 1 
                pending_signal = None
                continue

            # ==============================================================
            # ONNX NATIVE NEURAL INFERENCE (Triggered Only on 15m Close)
            # ==============================================================
            if latest_15m_features is not None:
                bars_since_last_trade += 1
                
                if trading_locked_for_day:
                    continue

                if len(self.feature_buffer) == 30:
                    
                    # 1. Oracle Prediction
                    window_tensor = np.array(self.feature_buffer, dtype=np.float32)[np.newaxis, ...]
                    oracle_inputs = {self.oracle_session.get_inputs()[0].name: window_tensor}
                    
                    # Run Oracle ONNX Graph
                    logits = self.oracle_session.run(None, oracle_inputs)[0]
                    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
                    prob_hold, prob_long, prob_short = probs[0][0], probs[0][1], probs[0][2]

                    # 2. Re-inject real-time probabilities into the feature vector
                    # Indices 36, 37, 38 align with the feature_cols map
                    feature_vector[36] = prob_long
                    feature_vector[37] = prob_short
                    feature_vector[38] = prob_hold

                    EXECUTION_THRESHOLD = 0.35
                    current_h4_trend = latest_15m_features.get("h4_trend", 0)
                    env_atr = latest_15m_features.get("env_atr", 1.0)
                    direction = 0

                    if prob_long > EXECUTION_THRESHOLD and prob_long > prob_short and current_h4_trend > 0:
                        direction = 1
                    elif prob_short > EXECUTION_THRESHOLD and prob_short > prob_long and current_h4_trend < 0:
                        direction = 2

                    if direction != 0:
                        if bars_since_last_trade < 4 or trades_today >= self.max_trades_per_day: 
                            direction = 0  

                    if direction != 0:
                        # 3. SAC Manager Prediction (42 Dimensions)
                        obs = np.zeros(42, dtype=np.float32)
                        
                        obs[:39] = feature_vector
                        obs[39] = float(np.clip(equity / self.initial_balance, 0.0, 10.0))
                        obs[40] = float(np.clip((high_water_mark - equity) / high_water_mark, 0.0, 1.0))
                        obs[41] = float(np.clip(bars_since_last_trade / 480.0, 0.0, 1.0))
                        
                        obs_input = {self.manager_session.get_inputs()[0].name: obs[np.newaxis, ...]}
                        
                        # Run SAC ONNX Graph
                        action = self.manager_session.run(None, obs_input)[0][0]
                        
                        size_val, tp_val, sl_val = action[0], action[1], action[2]
                        
                        sl_mult = ((sl_val + 1.0) / 2.0) * 1.0 + 0.5
                        tp_mult = sl_mult * (((tp_val + 1.0) / 2.0) * 2.0 + 1.0)
                        
                        pending_signal = {
                            "type": "Long" if direction == 1 else "Short",
                            "sl_distance": (env_atr * sl_mult) * 10,
                            "tp_distance": (env_atr * tp_mult) * 10
                        }

        # Print Final Report
        yield_df = pd.DataFrame(challenge_history)
        print("\n" + "=" * 60)
        print(" ONNX SIMULATOR CHALLENGE REPORT ")
        print("=" * 60)
        if not yield_df.empty:
            print(f"Challenges PASSED: {len(yield_df[yield_df['Result'] == 'PASSED'])}")
        return yield_df

if __name__ == "__main__":
    # Ensure these paths point to the deployed .onnx files downloaded from Colab
    XAU = "data/XAUUSDr_M1_OG.csv"
    DXY = "data/USDIndex_M1_OG.csv"
    ORACLE_ONNX = "deployed/oracle_v3.onnx"
    MANAGER_ONNX = "deployed/manager_actor_v3.onnx"
    
    sim = M1HighFidelitySimulator(XAU, DXY, ORACLE_ONNX, MANAGER_ONNX)
    sim.run_simulation()