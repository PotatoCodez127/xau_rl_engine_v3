import os
import torch
import numpy as np
import pandas as pd
from datetime import timezone
from stable_baselines3 import SAC
from models.oracle.attention_net import TemporalAttentionOracle

class OmniVolatilityGatekeeper:
    """Pre-trade firewall to physically block execution in structural chop zones."""
    def __init__(self, min_volatility_percentile=0.45):
        # 0.45 means the current H1 range must be larger than 45% of the last 24 hours.
        self.min_volatility_percentile = min_volatility_percentile

    def authorize_execution(self, h1_vol_regime: float, h4_trend: float, oracle_direction: int) -> bool:
        # Block 1: Liquidity Void Check (Is the market moving enough to hit 2R?)
        if h1_vol_regime < self.min_volatility_percentile:
            return False
            
        # Block 2: Macro Trend Alignment (Is the 15m momentum fighting the 4H river?)
        if oracle_direction == 1 and h4_trend < 0:
            return False
        if oracle_direction == 2 and h4_trend > 0:
            return False
            
        return True

class HighFidelitySimulator:
    def __init__(self, data_path, oracle_path, manager_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Initializing High-Fidelity V3.2 Simulator for Prop Firm Execution...")

        # Load Data
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Ensure UTC timezone alignment for CI runner temporal agreement
        if self.df.index.tz is None:
            self.df.index = self.df.index.tz_localize('UTC')
        else:
            self.df.index = self.df.index.tz_convert('UTC')

        # Broker & Risk Constants
        self.commission_per_lot = 5.00
        self.spread_pips = 2.0
        self.pip_value_per_lot = 10.0

        # --- NEW: Dynamic Compounding ---
        self.risk_pct = 0.015  # Risking exactly 1.5% of current equity per trade
        self.initial_balance = 10000.0

        # System Constants (Aligned with xau_dynamic_env.py)
        self.min_bars_between_trades = 48  # Max 2 trade per day

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
        # Daily Rollover (23:45 to 00:30)
        if (current_time.hour == 23 and current_time.minute >= 45) or (
            current_time.hour == 0 and current_time.minute <= 30
        ):
            return True
        return False

    def run_simulation(self):
        holdout_start_idx = int(len(self.df) * 0.8)
        print(f"Executing Asynchronous Backtest Engine...")
        print(f"Enforcing OOS Firewall: Starting simulation at step {holdout_start_idx}.")

        # Instantiate the Gatekeeper
        gatekeeper = OmniVolatilityGatekeeper(min_volatility_percentile=0.45)

        equity = self.initial_balance
        high_water_mark = self.initial_balance
        daily_start_equity = self.initial_balance
        current_day = self.df.index[holdout_start_idx].date()
        
        trading_locked_for_day = False
        account_failed = False
        account_passed = False

        equity_curve = [equity]
        journal = []
        active_trade = None
        pending_signal = None
        bars_since_last_trade = 0

        for i in range(holdout_start_idx, len(self.df) - 1):
            current_time = self.df.index[i]
            current_bar = self.df.iloc[i]
            bars_since_last_trade += 1

            # --- UTC Temporal Synchronization for Daily Reset ---
            if current_time.date() > current_day:
                daily_start_equity = equity
                current_day = current_time.date()
                trading_locked_for_day = False

            # --- 1. NON-BLOCKING PRICE TRACKING (Manage Active Trade) ---
            if active_trade is not None:
                trade_closed = False
                exit_price = 0.0
                exit_reason = ""

                # Calculate real-time floating PnL (rough midpoint estimate for tick loop)
                current_price = current_bar["env_open"]
                current_distance_pips = (current_price - active_trade["entry"]) * 10.0
                if active_trade["type"] == "Short":
                    current_distance_pips = -current_distance_pips
                
                floating_pnl = current_distance_pips * self.pip_value_per_lot * active_trade["lot_size"] - active_trade["total_friction"]
                current_floating_equity = equity + floating_pnl

                # Global Circuit Breakers (Floating Status)
                if current_floating_equity <= (daily_start_equity - self.max_daily_loss):
                    trade_closed, exit_price, exit_reason = (True, current_price, "Daily Drawdown Breached")
                    trading_locked_for_day = True
                elif current_floating_equity <= (high_water_mark - self.max_trailing_loss):
                    trade_closed, exit_price, exit_reason = (True, current_price, "Trailing Drawdown Breached")
                    account_failed = True
                elif current_floating_equity >= self.profit_target:
                    trade_closed, exit_price, exit_reason = (True, current_price, "Profit Target Reached")
                    account_passed = True
                
                # Rule Overrides
                elif floating_pnl <= self.guardian_shield_loss:
                    trade_closed, exit_price, exit_reason = (True, current_price, "Guardian Shield (1% Loss)")
                elif floating_pnl >= self.consistency_cap_usd:
                    trade_closed, exit_price, exit_reason = (True, current_price, "Consistency Hard Clip (15% Rule)")
                
                # Standard Target Executions
                elif active_trade["type"] == "Long":
                    if current_bar["env_low"] <= active_trade["sl"]:
                        trade_closed, exit_price, exit_reason = (True, active_trade["sl"], "Stop Loss")
                    elif current_bar["env_high"] >= active_trade["tp"]:
                        trade_closed, exit_price, exit_reason = (True, active_trade["tp"], "Take Profit")
                elif active_trade["type"] == "Short":
                    if current_bar["env_high"] >= active_trade["sl"]:
                        trade_closed, exit_price, exit_reason = (True, active_trade["sl"], "Stop Loss")
                    elif current_bar["env_low"] <= active_trade["tp"]:
                        trade_closed, exit_price, exit_reason = (True, active_trade["tp"], "Take Profit")

                if trade_closed:
                    pip_diff = (exit_price - active_trade["entry"]) * 10
                    if active_trade["type"] == "Short":
                        pip_diff *= -1

                    gross_pnl = (pip_diff * self.pip_value_per_lot * active_trade["lot_size"])
                    net_pnl = gross_pnl - active_trade["total_friction"]
                    equity += net_pnl

                    if equity > high_water_mark:
                        high_water_mark = equity

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

            # Global Circuit Breakers (Idle Status)
            if active_trade is None:
                if equity <= (daily_start_equity - self.max_daily_loss):
                    trading_locked_for_day = True
                if equity <= (high_water_mark - self.max_trailing_loss):
                    account_failed = True
                if equity >= self.profit_target:
                    account_passed = True

            if account_failed or account_passed:
                break

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

                # --- NEW: Regime-Modulated Fractional Half-Kelly ---
                # 1. Expected Winrate (Oracle Conviction blended with OOS Baseline)
                wfa_baseline_winrate = 0.3622
                p = (pending_signal["prob_win"] + wfa_baseline_winrate) / 2.0
                
                # 2. Odds (Reward-to-Risk)
                b = pending_signal["reward_to_risk"]

                # 3. Kelly Criterion Formula
                kelly_fraction = p - ((1.0 - p) / b) if b > 0 else 0.0
                
                # 4. Fractional Half-Kelly Floor
                # Floor at 0.1% to allow the agent to execute low-conviction structural trades without risking ruin
                half_kelly = max(kelly_fraction / 2.0, 0.001) 
                
                # 5. Volatility Regime Modulation
                # Throttle sizing by up to 25% during violent H1 volatility percentiles (protect against macro sweeps)
                regime_scalar = 1.0 - (pending_signal["h1_vol_regime"] * 0.25)
                
                final_risk_pct = half_kelly * regime_scalar
                
                # Ceiling at 5% to prevent catastrophic failure on model hallucination
                final_risk_pct = np.clip(final_risk_pct, 0.001, 0.05)
                
                current_risk_usd = equity * final_risk_pct

                # Lot_Volume = Current_Risk_USD / (SL_Pips * Pip_Value)
                theoretical_lot_size = current_risk_usd / (sl_pips * self.pip_value_per_lot)
                
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
                bars_since_last_trade = 0
                pending_signal = None
                continue

            # --- 3. SIGNAL GENERATION ---
            if trading_locked_for_day or self.is_restricted_time(current_time):
                continue
            
            prob_hold, prob_long, prob_short = self._get_oracle_probs(i)
            EXECUTION_THRESHOLD = 0.35
            current_h4_trend = current_bar.get("h4_trend", 0)
            current_h1_vol_regime = current_bar.get("h1_vol_regime", 0.5)
            
            # Step A: Determine raw Oracle intent
            oracle_direction = 0
            if prob_long > EXECUTION_THRESHOLD and prob_long > prob_short:
                oracle_direction = 1
            elif prob_short > EXECUTION_THRESHOLD and prob_short > prob_long:
                oracle_direction = 2

            # Step B: Gatekeeper Authorization
            direction = 0
            if oracle_direction != 0:
                is_authorized = gatekeeper.authorize_execution(
                    h1_vol_regime=current_h1_vol_regime,
                    h4_trend=current_h4_trend,
                    oracle_direction=oracle_direction
                )
                
                # Step C: Temporal & Cooldown Check
                if is_authorized and bars_since_last_trade >= self.min_bars_between_trades:
                    direction = oracle_direction

            # Step D: SAC Manager Execution (Only triggered if direction != 0)
            if direction != 0:
                features = current_bar[self.feature_cols].values
                obs = np.zeros(len(self.feature_cols) + 6, dtype=np.float32)

            if direction != 0:
                features = current_bar[self.feature_cols].values
                obs = np.zeros(len(self.feature_cols) + 6, dtype=np.float32)
                obs[: len(features)] = features
                obs[len(features)] = prob_hold
                obs[len(features) + 1] = prob_long
                obs[len(features) + 2] = prob_short
                obs[-3] = float(np.clip(equity / self.initial_balance, 0.0, 10.0))
                obs[-2] = float(np.clip((high_water_mark - equity) / high_water_mark, 0.0, 1.0))
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
        print("   PROP FIRM SIMULATION REPORT (V3.2)   ")
        print("=" * 50)
        if not journal_df.empty:
            print(f"Total Trades Executed: {len(journal_df)}")
            print(f"Final Account Equity:  ${equity:.2f}")
            print(f"High Water Mark:       ${high_water_mark:.2f}")
            print(f"Account Passed:        {account_passed}")
            print(f"Account Failed:        {account_failed}")
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