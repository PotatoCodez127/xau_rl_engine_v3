import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class XAUDynamicEnv(gym.Env):
    """
    XAUUSD Master-Slave Execution Environment (V3.2).
    Phase A (Oracle) strictly dictates direction. Phase B (SAC Manager) exclusively dictates sizing.
    Refactored: Dynamic MFE Exits, Symmetry Forcing, and Risk-Adjusted Reward Shaping.
    """
    def __init__(self, df: pd.DataFrame, initial_balance=10000.0):
        super(XAUDynamicEnv, self).__init__()
        
        self.df = df.reset_index(drop=True) 
        self.initial_balance = initial_balance
        self.friction_cost = 10.0

        self.min_bars_between_trades = 4 
        self.max_trades_per_day = 5       
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        exclude_cols = ["target", "time", "datetime", "date", "index"]
        self.feature_cols = [
            c for c in self.df.columns if c not in exclude_cols and not c.startswith("env_")
        ]

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(len(self.feature_cols) + 3,), dtype=np.float32
        )

        # AOT Memory Allocation
        self.features_mat = np.ascontiguousarray(self.df[self.feature_cols].values, dtype=np.float32)
        self.dates_arr = pd.to_datetime(self.df["datetime"]).dt.date.values
        self.prob_long_arr = np.ascontiguousarray(self.df["prob_long"].values, dtype=np.float32)
        self.prob_short_arr = np.ascontiguousarray(self.df["prob_short"].values, dtype=np.float32)
        self.prob_hold_arr = np.ascontiguousarray(self.df["prob_hold"].values, dtype=np.float32)
        self.h4_trend_arr = np.ascontiguousarray(self.df["h4_trend"].values, dtype=np.float32)
        
        self.max_steps = len(self.features_mat) - 1

        # State Variables
        self.current_step = 30
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.bars_since_last_trade = 0
        self.trades_today = 0
        self.current_day = self.dates_arr[self.current_step]
        
        # --- NEW: Symmetry Forcing History ---
        self.trade_history = deque(maxlen=20)
        self.long_count = 0
        self.short_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 30
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.bars_since_last_trade = 0
        self.trades_today = 0
        self.current_day = self.dates_arr[self.current_step]
        
        # Reset symmetry tracking
        self.trade_history.clear()
        self.long_count = 0
        self.short_count = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        X = self.features_mat[self.current_step]
        obs[: len(X)] = X

        obs[-3] = float(np.clip(self.balance / self.initial_balance, 0.0, 10.0))
        obs[-2] = float(np.clip((self.peak_balance - self.balance) / self.peak_balance, 0.0, 1.0))
        obs[-1] = float(np.clip(self.bars_since_last_trade / 480.0, 0.0, 1.0))
        
        return np.clip(obs, -10.0, 10.0)

    def _scale_action(self, action: np.ndarray) -> tuple[float, float]:
        sl_mult = 0.5 + ((action[0] + 1.0) * (2.0 - 0.5)) / 2.0
        tp_mult = 1.0 + ((action[1] + 1.0) * (3.0 - 1.0)) / 2.0
        return sl_mult, tp_mult

    def _evaluate_master_slave_trigger(self, prob_long: float, prob_short: float, prob_hold: float, h4_trend: float) -> int:
        EXECUTION_THRESHOLD = 0.40
        if prob_long > EXECUTION_THRESHOLD and prob_long > prob_hold and h4_trend > 0:
            return 1
        elif prob_short > EXECUTION_THRESHOLD and prob_short > prob_hold and h4_trend < 0:
            return 2
        return 0

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
                symmetry_penalty = (imbalance_ratio - 0.8) * 5.0 
            
            amount_at_risk = self.balance * 0.015
            prob_win = prob_long if direction == 1 else prob_short

            # --- DYNAMIC EXITS ---
            is_win = np.random.rand() < prob_win
            
            if is_win:
                mfe_haircut = np.clip(1.0 - (tp_mult_used * 0.05), 0.5, 1.0)
                simulated_pnl = (amount_at_risk * (tp_mult_used / sl_mult_used)) * mfe_haircut
            else:
                simulated_pnl = -amount_at_risk * np.clip(sl_mult_used, 0.5, 1.0)

            simulated_pnl -= self.friction_cost
            self.balance += simulated_pnl
            self.peak_balance = max(max(self.initial_balance, self.peak_balance), self.balance)
            
            # --- RISK-ADJUSTED REWARD RESHAPING ---
            drawdown = (self.peak_balance - self.balance) / self.peak_balance
            
            # Sortino Proxy: Penalize high action variance
            sizing_risk_penalty = ((sl_mult_used - 0.5) / 1.5) * (self.initial_balance * 0.015)
            drawdown_penalty = drawdown * self.initial_balance * 0.5 
            
            raw_reward = simulated_pnl - drawdown_penalty - sizing_risk_penalty
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