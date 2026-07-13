import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class XAUDynamicEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, initial_balance=10000.0):
        # ... [Keep existing init code] ...
        
        # Symmetry Forcing: Track last 20 trade directions
        self.trade_history = deque(maxlen=20)
        self.long_count = 0
        self.short_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # ... [Keep existing reset variables] ...
        self.trade_history.clear()
        self.long_count = 0
        self.short_count = 0
        return self._get_obs(), {}

    def step(self, action):
        step_date = self.dates_arr[self.current_step]
        if step_date > self.current_day:
            self.current_day = step_date
            self.trades_today = 0

        sl_mult_used, tp_mult_ratio = self._scale_action(action)
        tp_mult_used = sl_mult_used * tp_mult_ratio 

        prob_long = self.prob_long_arr[self.current_step]
        prob_short = self.prob_short_arr[self.current_step]
        prob_hold = self.prob_hold_arr[self.current_step]
        h4_trend = self.h4_trend_arr[self.current_step]
        
        direction = self._evaluate_master_slave_trigger(prob_long, prob_short, prob_hold, h4_trend)

        if self.bars_since_last_trade < self.min_bars_between_trades or self.trades_today >= self.max_trades_per_day:
            direction = 0

        simulated_pnl = 0.0
        self.bars_since_last_trade += 1
        reward = 0.0  

        if direction != 0:
            self.bars_since_last_trade = 0
            self.trades_today += 1
            
            # --- SYMMETRY FORCING ---
            self.trade_history.append(direction)
            self.long_count = sum(1 for d in self.trade_history if d == 1)
            self.short_count = sum(1 for d in self.trade_history if d == 2)
            
            imbalance_ratio = max(self.long_count, self.short_count) / max(1, len(self.trade_history))
            symmetry_penalty = 0.0
            if len(self.trade_history) >= 10 and imbalance_ratio > 0.8:
                # Penalize heavy unidirectional bias
                symmetry_penalty = (imbalance_ratio - 0.8) * 5.0 
            
            amount_at_risk = self.balance * 0.015
            prob_win = prob_long if direction == 1 else prob_short

            # --- DYNAMIC EXITS & EXCURSION PROXY ---
            # Instead of a binary win/loss, we model MFE/MAE probability based on action sizing
            is_win = np.random.rand() < prob_win
            
            if is_win:
                # MFE Protection: Larger TP targets have higher probability of reversing before fill.
                # We dynamically haircut the profit based on how greedy the TP multiplier is.
                mfe_haircut = np.clip(1.0 - (tp_mult_used * 0.05), 0.5, 1.0)
                simulated_pnl = (amount_at_risk * (tp_mult_used / sl_mult_used)) * mfe_haircut
            else:
                # MAE Cutting: Tighter stop losses cut the trade earlier, saving capital.
                # A wide SL multiplier results in full loss.
                simulated_pnl = -amount_at_risk * np.clip(sl_mult_used, 0.5, 1.0)

            simulated_pnl -= self.friction_cost
            self.balance += simulated_pnl
            self.peak_balance = max(max(self.initial_balance, self.peak_balance), self.balance)
            
            # --- RISK-ADJUSTED REWARD RESHAPING ---
            drawdown = (self.peak_balance - self.balance) / self.peak_balance
            
            # Sortino/Risk Proxy: Penalize the sizing directly. 
            # High action variance (sl_mult_used near 2.0) incurs a stability tax.
            sizing_risk_penalty = ((sl_mult_used - 0.5) / 1.5) * (self.initial_balance * 0.005)
            drawdown_penalty = drawdown * self.initial_balance * 0.5 # Increased DB penalty
            
            raw_reward = simulated_pnl - drawdown_penalty - sizing_risk_penalty
            
            # Normalize and apply symmetry penalty
            normalized_reward = float(np.clip((raw_reward / (self.initial_balance * 0.01)), -10.0, 10.0))
            reward = normalized_reward - symmetry_penalty

        self.current_step += 1
        terminated = self.balance < (self.initial_balance * 0.1)
        truncated = self.current_step >= self.max_steps

        info = {
            "prob_long": prob_long,
            "prob_short": prob_short,
            "sl_mult_used": sl_mult_used,
            "tp_mult_used": tp_mult_used,
            "imbalance_ratio": getattr(self, 'long_count', 0) / max(1, len(self.trade_history))
        }

        return self._get_obs(), reward, terminated, truncated, info