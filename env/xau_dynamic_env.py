import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class XAUDynamicEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, initial_balance=10000.0):
        super(XAUDynamicEnv, self).__init__()
        # Preserve datetime for UTC session boundaries
        self.df = df.reset_index()
        self.initial_balance = initial_balance
        self.friction_cost = 10.0

        self.min_bars_between_trades = 4  # Aligned with OOS
        self.max_trades_per_day = 1
        self.urgency_threshold = 240  

        # ACTION SPACE REDUCED TO 3 (Size, TP, SL). Direction is handled by the Oracle.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        exclude_cols = ["target", "time", "datetime", "date", "index"]
        self.feature_cols = [
            c for c in df.columns if c not in exclude_cols and not c.startswith("env_")
        ]

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(len(self.feature_cols) + 3,), dtype=np.float32
        )

        self.current_step = 30
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.bars_since_last_trade = 0
        self.trades_today = 0
        self.current_day = pd.to_datetime(self.df.loc[self.current_step, "datetime"]).date()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 30
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.bars_since_last_trade = 0
        self.trades_today = 0
        self.current_day = pd.to_datetime(self.df.loc[self.current_step, "datetime"]).date()
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        X = self.df.loc[self.current_step, self.feature_cols].values
        obs[: len(X)] = X

        obs[-3] = float(np.clip(self.balance / self.initial_balance, 0.0, 10.0))
        obs[-2] = float(np.clip((self.peak_balance - self.balance) / self.peak_balance, 0.0, 1.0))
        obs[-1] = float(np.clip(self.bars_since_last_trade / 480.0, 0.0, 1.0))
        return np.clip(obs, -10.0, 10.0)

    def step(self, action):
        step_time = pd.to_datetime(self.df.loc[self.current_step, "datetime"])
        
        # Automated CI Runner Temporal Synchronization (UTC)
        if step_time.date() > self.current_day:
            self.current_day = step_time.date()
            self.trades_today = 0

        # Unpack the 3D action space
        size_val, tp_val, sl_val = action[0], action[1], action[2]

        # Oracle Master-Slave Logic
        prob_long = self.df.loc[self.current_step, "prob_long"]
        prob_short = self.df.loc[self.current_step, "prob_short"]
        h4_trend = self.df.loc[self.current_step, "h4_trend"]
        
        direction = 0
        EXECUTION_THRESHOLD = 0.35
        if prob_long > EXECUTION_THRESHOLD and prob_long > prob_short and h4_trend > 0:
            direction = 1
        elif prob_short > EXECUTION_THRESHOLD and prob_short > prob_long and h4_trend < 0:
            direction = 2

        # Constraints
        if self.bars_since_last_trade < self.min_bars_between_trades or self.trades_today >= self.max_trades_per_day:
            direction = 0

        simulated_pnl = 0.0
        frequency_penalty = 0.0
        self.bars_since_last_trade += 1

        if direction != 0:
            self.bars_since_last_trade = 0
            self.trades_today += 1
            
            risk_pct = ((size_val + 1.0) / 2.0) * 0.05
            sl_mult_used = ((sl_val + 1.0) / 2.0) * 1.0 + 0.5
            
            # Floor TP at 1.0x, Ceiling at 3.0x to avoid baseline inversion
            tp_mult_used = sl_mult_used * (((tp_val + 1.0) / 2.0) * 2.0 + 1.0) 
            amount_at_risk = self.balance * risk_pct

            prob_win = prob_long if direction == 1 else prob_short

            if np.random.rand() < prob_win:
                simulated_pnl = amount_at_risk * (tp_mult_used / sl_mult_used)
            else:
                simulated_pnl = -amount_at_risk

            simulated_pnl -= self.friction_cost
        else:
            if self.bars_since_last_trade > self.urgency_threshold:
                frequency_penalty = -0.05 * ((self.bars_since_last_trade - self.urgency_threshold) / 100.0)

        self.balance += simulated_pnl
        self.peak_balance = max(max(self.initial_balance, self.peak_balance), self.balance)
        drawdown = (self.peak_balance - self.balance) / self.peak_balance

        # Removed the arbitrary rr_penalty to allow 1.5R momentum scalps
        raw_reward = simulated_pnl - (drawdown * self.initial_balance * 0.1)
        reward = float(np.clip((raw_reward / (self.initial_balance * 0.01)) + frequency_penalty, -10.0, 10.0))

        self.current_step += 1
        terminated = self.balance < (self.initial_balance * 0.1)
        truncated = self.current_step >= len(self.df) - 1

        return self._get_obs(), reward, terminated, truncated, {}