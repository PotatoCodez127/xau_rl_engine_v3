import os
import time
import numpy as np
import pandas as pd
from datetime import timezone
import onnxruntime as ort

class DualM1DataFeed:
    """Synchronized dual-stream M1 feed for multi-symbol historical testing."""
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
    """Stateful 15-minute indicator feature builder."""
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

        return self._calculate_current_features() if self.is_warmed_up else None

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

        df['prob_long'] = 0.0
        df['prob_short'] = 0.0
        df['prob_hold'] = 1.0

        return df.iloc[-1].to_dict()


class DeepDiagnosticsSimulator:
    def __init__(self, xau_path, dxy_path, oracle_onnx_path, manager_onnx_path):
        print("Initializing Deep Diagnostics ONNX Engine...")
        self.feed = DualM1DataFeed(xau_path, dxy_path)
        self.engine = StreamingFeatureEngine(window_size=1000)

        self.commission_per_lot = 5.00
        self.spread_pips = 2.0
        self.pip_value_per_lot = 10.0
        self.initial_balance = 10000.0
        self.fixed_risk_usd = 50.00

        self.oracle_session = ort.InferenceSession(oracle_onnx_path, providers=['CPUExecutionProvider'])
        self.manager_session = ort.InferenceSession(manager_onnx_path, providers=['CPUExecutionProvider'])
        
        from collections import deque
        self.feature_buffer = deque(maxlen=30)
        
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

    def run_backtest(self):
        total_ticks = len(self.feed.master_stream)
        holdout_start_idx = 3000 
        print("Running full capacity continuous OOS diagnostics...")

        equity = self.initial_balance
        high_water_mark = self.initial_balance
        max_drawdown = 0.0

        journal = []
        latency_ns_logs = []
        active_trade = None
        pending_signal = None
        bars_since_last_trade = 0

        for idx, (timestamp, tick_row) in enumerate(self.feed.stream()):
            latest_15m_features = self.engine.process_m1_tick(timestamp, tick_row)
            if latest_15m_features is not None:
                feature_vector = [latest_15m_features.get(c, 0.0) if not np.isnan(latest_15m_features.get(c, 0.0)) else 0.0 for c in self.feature_cols]
                self.feature_buffer.append(feature_vector)
            
            if idx < holdout_start_idx:
                continue

            # Telemetry tracking for active trade (MAE / MFE)
            if active_trade is not None:
                if active_trade["type"] == "Long":
                    active_trade["max_favorable"] = max(active_trade["max_favorable"], tick_row["xau_high"] - active_trade["entry"])
                    active_trade["max_adverse"] = min(active_trade["max_adverse"], tick_row["xau_low"] - active_trade["entry"])
                    worst_price = tick_row["xau_low"]
                    best_price = tick_row["xau_high"]
                else:
                    active_trade["max_favorable"] = max(active_trade["max_favorable"], active_trade["entry"] - tick_row["xau_low"])
                    active_trade["max_adverse"] = min(active_trade["max_adverse"], active_trade["entry"] - tick_row["xau_high"])
                    worst_price = tick_row["xau_high"] + (self.spread_pips * 0.1)
                    best_price = tick_row["xau_low"]

                trade_closed = False
                exit_price = 0.0
                exit_reason = ""

                if active_trade["type"] == "Long":
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
                    if active_trade["type"] == "Short": pip_diff *= -1
                    gross_pnl = pip_diff * self.pip_value_per_lot * active_trade["lot_size"]
                    net_pnl = gross_pnl - active_trade["total_friction"]
                    equity += net_pnl

                    if equity > high_water_mark:
                        high_water_mark = equity
                    dd = high_water_mark - equity
                    if dd > max_drawdown: max_drawdown = dd

                    journal.append({
                        "Entry_Time": active_trade["time"],
                        "Exit_Time": timestamp,
                        "Type": active_trade["type"],
                        "Net_PnL": round(net_pnl, 2),
                        "MAE_Pips": round(active_trade["max_adverse"] * 10, 2),
                        "MFE_Pips": round(active_trade["max_favorable"] * 10, 2),
                        "Oracle_Long_Conf": active_trade["conf_long"],
                        "SAC_Sizing_Action": active_trade["sac_action_val"],
                        "Reason": exit_reason
                    })
                    active_trade = None
                    continue

            # Queue Processor
            if pending_signal is not None and active_trade is None:
                fill_price = tick_row["xau_open"]
                sl_dist = max(pending_signal["sl_distance"], 10.0)
                lot_size = np.clip(round(self.fixed_risk_usd / (sl_dist * self.pip_value_per_lot), 2), 0.01, 100.0)
                friction = (lot_size * self.commission_per_lot) + (lot_size * self.spread_pips * self.pip_value_per_lot)
                
                active_trade = {
                    "time": timestamp,
                    "type": pending_signal["type"],
                    "entry": fill_price,
                    "sl": fill_price - (sl_dist * 0.1) if pending_signal["type"] == "Long" else fill_price + (sl_dist * 0.1),
                    "tp": fill_price + (pending_signal["tp_distance"] * 0.1) if pending_signal["type"] == "Long" else fill_price - (pending_signal["tp_distance"] * 0.1),
                    "lot_size": lot_size,
                    "total_friction": friction,
                    "max_favorable": 0.0,
                    "max_adverse": 0.0,
                    "conf_long": pending_signal["conf_long"],
                    "sac_action_val": pending_signal["sac_action_val"]
                }
                bars_since_last_trade = 0
                pending_signal = None
                continue

            # Inference Loop on 15m Close
            if latest_15m_features is not None:
                bars_since_last_trade += 1
                if len(self.feature_buffer) == 30:
                    t_start = time.perf_counter_ns()
                    
                    window_tensor = np.array(self.feature_buffer, dtype=np.float32)[np.newaxis, ...]
                    logits = self.oracle_session.run(None, {self.oracle_session.get_inputs()[0].name: window_tensor})[0]
                    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
                    prob_hold, prob_long, prob_short = probs[0][0], probs[0][1], probs[0][2]

                    feature_vector[36] = prob_long
                    feature_vector[37] = prob_short
                    feature_vector[38] = prob_hold

                    direction = 0
                    if prob_long > 0.35 and prob_long > prob_short and latest_15m_features.get("h4_trend", 0) > 0:
                        direction = 1
                    elif prob_short > 0.35 and prob_short > prob_long and latest_15m_features.get("h4_trend", 0) < 0:
                        direction = 2

                    if direction != 0 and bars_since_last_trade >= 4:
                        obs = np.zeros(42, dtype=np.float32)
                        obs[:39] = feature_vector
                        obs[39] = float(np.clip(equity / self.initial_balance, 0.0, 10.0))
                        obs[40] = float(np.clip((high_water_mark - equity) / high_water_mark, 0.0, 1.0))
                        obs[41] = float(np.clip(bars_since_last_trade / 480.0, 0.0, 1.0))
                        
                        action = self.manager_session.run(None, {self.manager_session.get_inputs()[0].name: obs[np.newaxis, ...]})[0][0]
                        
                        # FIX: Unpack only 2 action dimensions matching XAUDynamicEnv._scale_action
                        sl_val, tp_val = action[0], action[1]
                        sl_mult = 0.5 + ((sl_val + 1.0) * (2.0 - 0.5)) / 2.0
                        tp_mult_ratio = 1.0 + ((tp_val + 1.0) * (3.0 - 1.0)) / 2.0
                        tp_mult = sl_mult * tp_mult_ratio
                        
                        env_atr = latest_15m_features.get("env_atr", 1.0)
                        
                        pending_signal = {
                            "type": "Long" if direction == 1 else "Short",
                            "sl_distance": (env_atr * sl_mult) * 10,
                            "tp_distance": (env_atr * tp_mult) * 10,
                            "conf_long": round(prob_long, 4),
                            "sac_action_val": round(sl_val, 4)
                        }

                    t_end = time.perf_counter_ns()
                    latency_ns_logs.append(t_end - t_start)

        # Tear Sheet Computation
        journal_df = pd.DataFrame(journal)
        print("\n" + "="*60)
        print(" INSTITUTIONAL PERFORMANCE TEAR SHEET ")
        print("="*60)
        if not journal_df.empty:
            wins = journal_df[journal_df["Net_PnL"] > 0]
            losses = journal_df[journal_df["Net_PnL"] <= 0]
            winrate = len(wins) / len(journal_df) * 100
            profit_factor = wins["Net_PnL"].sum() / abs(losses["Net_PnL"].sum()) if len(losses) > 0 else np.inf
            mean_latency_ms = np.mean(latency_ns_logs) / 1_000_000
            
            print(f"Total Trades Taken:      {len(journal_df)}")
            print(f"Strategy Winrate:        {winrate:.2f}%")
            print(f"Profit Factor:           {profit_factor:.2f}")
            print(f"Max Capital Drawdown:    ${max_drawdown:.2f}")
            print(f"Mean Inference Latency:  {mean_latency_ms:.3f} ms")
            os.makedirs("logs", exist_ok=True)
            journal_df.to_csv("logs/deep_diagnostics_report.csv", index=False)
            print("💾 Saved full telemetry to logs/deep_diagnostics_report.csv")
        print("="*60)

if __name__ == "__main__":
    XAU = "data/raw/XAUUSDr_M1_OG.csv"
    DXY = "data/raw/USDIndex_M1_OG.csv"
    ORACLE_ONNX = "models/deployed/oracle_v3.onnx"
    MANAGER_ONNX = "models/deployed/manager_actor_v3.onnx"
    
    sim = DeepDiagnosticsSimulator(XAU, DXY, ORACLE_ONNX, MANAGER_ONNX)
    sim.run_backtest()