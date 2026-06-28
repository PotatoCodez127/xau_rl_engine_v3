import os
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from models.oracle.attention_net import TemporalAttentionOracle


class HighFidelitySimulator:
    def __init__(self, data_path, oracle_path, manager_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Initializing High-Fidelity V3.2 Simulator for Prop Firm Execution...")

        # Load Data
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)

        # Broker & Risk Constants
        self.commission_per_lot = 5.00
        self.spread_pips = 2.0
        self.pip_value_per_lot = 10.0

        # --- PROP FIRM LIMITS ---
        self.initial_balance = 5000.0
        self.fixed_risk_usd = 25.00
        self.consistency_cap_usd = 37.50

        # System Constants (Aligned with xau_dynamic_env.py)
        self.min_bars_between_trades = 96  # Max 1 trade per day

        # Load AI Models
        self._load_models(oracle_path, manager_path)

    def _load_models(self, oracle_path, manager_path):
        exclude_cols = ["target", "time", "datetime", "date"]
        self.feature_cols = [
            c
            for c in self.df.columns
            if c not in exclude_cols and not c.startswith("env_")
        ]

        # Phase A: Oracle
        self.oracle = TemporalAttentionOracle(
            input_dim=len(self.feature_cols), seq_len=30
        ).to(self.device)
        self.oracle.load_state_dict(torch.load(oracle_path, map_location=self.device))
        self.oracle.eval()

        # Phase B: Manager (Load purely for inference)
        self.manager = SAC.load(manager_path, device=self.device)

    def _get_oracle_probs(self, current_step):
        window = (
            self.df[self.feature_cols].iloc[current_step - 30 : current_step].values
        )
        window_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.oracle(window_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return probs[0], probs[1], probs[2]

    def is_restricted_time(self, current_time: pd.Timestamp) -> bool:
        # 1. Daily Rollover (23:45 to 00:30)
        if (current_time.hour == 23 and current_time.minute >= 45) or (
            current_time.hour == 0 and current_time.minute <= 30
        ):
            return True
        # 2. Friday Close (No entries after 21:00 Friday)
        if current_time.weekday() == 4 and current_time.hour >= 21:
            return True
        return False

    def run_simulation(self):
        holdout_start_idx = int(len(self.df) * 0.8)
        print(f"Executing Asynchronous Backtest Engine...")
        print(f"Enforcing OOS Firewall: Starting simulation at step {holdout_start_idx}.")

        equity = self.initial_balance
        peak_equity = equity
        equity_curve = [equity]
        journal = []

        active_trade = None
        pending_signal = None
        bars_since_last_trade = 0

        for i in range(holdout_start_idx, len(self.df) - 1):
            current_time = self.df.index[i]
            current_bar = self.df.iloc[i]

            bars_since_last_trade += 1

            # --- 1. NON-BLOCKING PRICE TRACKING (Manage Active Trade) ---
            if active_trade is not None:
                trade_closed = False
                exit_price = 0.0
                exit_reason = ""

                # Weekend Liquidation Check
                if (
                    current_time.weekday() == 4
                    and current_time.hour >= 22
                    and current_time.minute >= 45
                ):
                    trade_closed = True
                    exit_price = current_bar["env_close"]
                    exit_reason = "Weekend Liquidation"
                else:
                    # Calculate real-time floating PnL (rough midpoint estimate for tick loop)
                    current_price = current_bar["env_open"]
                    current_distance_pips = (current_price - active_trade["entry"]) * 10.0
                    if active_trade["type"] == "Short":
                        current_distance_pips = -current_distance_pips
                    
                    floating_pnl = current_distance_pips * self.pip_value_per_lot * active_trade["lot_size"] - active_trade["total_friction"]

                    # 1. The Blind Consistency Clip (Simulator Override)
                    if floating_pnl >= self.consistency_cap_usd:
                        trade_closed, exit_price, exit_reason = (True, current_price, "Consistency Hard Clip")
                    
                    elif active_trade["type"] == "Long":
                        if current_bar["env_low"] <= active_trade["sl"]:
                            trade_closed, exit_price, exit_reason = (
                                True, active_trade["sl"], "Stop Loss"
                            )
                        elif current_bar["env_high"] >= active_trade["tp"]:
                            trade_closed, exit_price, exit_reason = (
                                True, active_trade["tp"], "Take Profit"
                            )

                    elif active_trade["type"] == "Short":
                        if current_bar["env_high"] >= active_trade["sl"]:
                            trade_closed, exit_price, exit_reason = (
                                True, active_trade["sl"], "Stop Loss"
                            )
                        elif current_bar["env_low"] <= active_trade["tp"]:
                            trade_closed, exit_price, exit_reason = (
                                True, active_trade["tp"], "Take Profit"
                            )

                if trade_closed:
                    pip_diff = (exit_price - active_trade["entry"]) * 10
                    if active_trade["type"] == "Short":
                        pip_diff *= -1

                    gross_pnl = (
                        pip_diff * self.pip_value_per_lot * active_trade["lot_size"]
                    )
                    net_pnl = gross_pnl - active_trade["total_friction"]

                    equity += net_pnl
                    if equity > peak_equity:
                        peak_equity = equity

                    journal.append(
                        {
                            "Entry_Time": active_trade["time"],
                            "Exit_Time": current_time,
                            "Type": active_trade["type"],
                            "Entry_Price": round(active_trade["entry"], 3),
                            "SL_Price": round(active_trade["sl"], 3),
                            "TP_Price": round(active_trade["tp"], 3),
                            "Exit_Price": round(exit_price, 3),
                            "Lot_Size": round(active_trade["lot_size"], 2),
                            "Friction_Cost": round(active_trade["total_friction"], 2),
                            "Net_PnL": round(net_pnl, 2),
                            "Equity": round(equity, 2),
                            "Reason": exit_reason,
                        }
                    )

                    active_trade = None
                    continue
            
            # --- 2. EXECUTE PENDING QUEUE (Latency Simulation $t$) ---
            if pending_signal is not None and active_trade is None:
                fill_price = current_bar["env_open"]
                atr = current_bar["env_atr"]
                slippage_pips = np.clip(atr * 0.1, 0.1, 1.5)

                if pending_signal["type"] == "Long":
                    fill_price += slippage_pips * 0.1
                else:
                    fill_price -= slippage_pips * 0.1

                sl_pips = pending_signal["sl_distance"]

                # --- PROP FIRM: CONSTANT DOLLAR RISK SIZING ---
                sl_distance_pips = sl_pips
                if sl_distance_pips < 10.0:
                    sl_distance_pips = 10.0

                # Dynamic Lot Sizing: Force every stop loss to equal exactly the fixed_risk_usd
                theoretical_lot_size = self.fixed_risk_usd / (sl_distance_pips * self.pip_value_per_lot)

                # Valid MetaTrader lot sizing: Min 0.01, Max 100.00, Step 0.01
                lot_size = round(theoretical_lot_size, 2)
                if lot_size < 0.01:
                    lot_size = 0.01
                lot_size = np.clip(lot_size, 0.01, 100.0)

                commission = lot_size * self.commission_per_lot
                spread_cost = lot_size * self.spread_pips * self.pip_value_per_lot
                total_friction = commission + spread_cost

                active_trade = {
                    "time": current_time,
                    "type": pending_signal["type"],
                    "entry": fill_price,
                    "sl": (
                        fill_price - (sl_pips * 0.1)
                        if pending_signal["type"] == "Long"
                        else fill_price + (sl_pips * 0.1)
                    ),
                    "tp": (
                        fill_price + (pending_signal["tp_distance"] * 0.1)
                        if pending_signal["type"] == "Long"
                        else fill_price - (pending_signal["tp_distance"] * 0.1)
                    ),
                    "lot_size": lot_size,
                    "total_friction": total_friction,
                }

                # Trade is officially executed. Reset the RL clock here.
                bars_since_last_trade = 0
                pending_signal = None
                continue

            # --- 3. SIGNAL GENERATION ---
            if self.is_restricted_time(current_time):
                continue

            prob_hold, prob_long, prob_short = self._get_oracle_probs(i)
            EXECUTION_THRESHOLD = 0.35
            current_h4_trend = current_bar.get("h4_trend", 0)

            direction = 0
            if prob_long > EXECUTION_THRESHOLD and prob_long > prob_short:
                if current_h4_trend > 0:
                    direction = 1
            elif prob_short > EXECUTION_THRESHOLD and prob_short > prob_long:
                if current_h4_trend < 0:
                    direction = 2

            if direction != 0 and bars_since_last_trade < self.min_bars_between_trades:
                direction = 0

            if direction != 0:
                features = current_bar[self.feature_cols].values
                obs = np.zeros(len(self.feature_cols) + 6, dtype=np.float32)
                obs[: len(features)] = features
                obs[len(features)] = prob_hold
                obs[len(features) + 1] = prob_long
                obs[len(features) + 2] = prob_short
                obs[-3] = float(np.clip(equity / self.initial_balance, 0.0, 10.0))
                obs[-2] = float(np.clip((peak_equity - equity) / peak_equity, 0.0, 1.0))
                obs[-1] = float(np.clip(bars_since_last_trade / 480.0, 0.0, 1.0))

                action, _ = self.manager.predict(obs, deterministic=True)
                size_val, tp_val, sl_val = action[1], action[2], action[3]

                sl_mult = ((sl_val + 1.0) / 2.0) * 1.0 + 0.5
                tp_mult = sl_mult * (((tp_val + 1.0) / 2.0) * 2.0 + 1.0)
                reward_to_risk = tp_mult / sl_mult

                pending_signal = {
                    "type": "Long" if direction == 1 else "Short",
                    "sl_distance": (current_bar["env_atr"] * sl_mult) * 10,
                    "tp_distance": (current_bar["env_atr"] * tp_mult) * 10,
                    "prob_win": prob_long if direction == 1 else prob_short,
                    "h1_vol_regime": current_bar.get("h1_vol_regime", 0.5),
                    "reward_to_risk": reward_to_risk
                }

            equity_curve.append(equity)

        # Print Final Report
        journal_df = pd.DataFrame(journal)
        print("\n" + "=" * 50)
        print(" 📡 PROP FIRM SIMULATION REPORT (V3.2) 📡")
        print("=" * 50)
        if not journal_df.empty:
            print(f"Total Trades Executed: {len(journal_df)}")
            print(f"Final Account Equity:  ${equity:.2f}")
            print(f"Average Lot Size:      {journal_df['Lot_Size'].mean():.2f} Lots")
            print(f"Average Friction/Trade:${journal_df['Friction_Cost'].mean():.2f}")
            wins = journal_df[journal_df["Net_PnL"] > 0]
            print(f"True Winrate:          {(len(wins)/len(journal_df))*100:.2f}%")
        else:
            print("No trades executed. Thresholds or temporal voids blocked all entries.")
        print("=" * 50)

        os.makedirs("logs", exist_ok=True)
        journal_df.to_csv("logs/high_fidelity_journal.csv", index=False)
        print("Detailed execution log saved to logs/high_fidelity_journal.csv")
        return journal_df


if __name__ == "__main__":
    DATA = "data/processed/labeled_features_15m.csv"
    ORACLE = "models/oracle/best_oracle.pth"
    MANAGER = "models/manager/saved/wfa_43/best_model.zip"

    sim = HighFidelitySimulator(DATA, ORACLE, MANAGER)
    sim.run_simulation()