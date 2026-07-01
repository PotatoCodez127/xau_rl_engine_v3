import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class XAUDynamicEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, initial_balance=10000.0):
        super(XAUDynamicEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.friction_cost = 10.0

        # --- NEW CONSTANTS FOR FREQUENCY ENFORCEMENT ---
        self.min_bars_between_trades = 96  # 1 Trade per Day maximum (96 bars of 15m)
        self.urgency_threshold = (
            240  # Start decaying reward after ~2.5 days without a trade
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        exclude_cols = ["target", "time", "datetime", "date"]
        self.feature_cols = [
            c for c in df.columns if c not in exclude_cols and not c.startswith("env_")
        ]

        # Increased observation space by 1 to include 'normalized_time_since_trade'
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(len(self.feature_cols) + 3,), dtype=np.float32
        )

        self.current_step = 30
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.bars_since_last_trade = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 30
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.bars_since_last_trade = 0
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        features = self.df.loc[self.current_step, self.feature_cols].values
        obs[: len(features)] = features

        equity_ratio = float(np.clip(self.balance / self.initial_balance, 0.0, 10.0))
        peak_equity = max(self.initial_balance, self.peak_balance)
        self.peak_balance = max(peak_equity, self.balance)
        drawdown = float(
            np.clip((self.peak_balance - self.balance) / self.peak_balance, 0.0, 1.0)
        )

        # Let the agent "see" its own frequency clock (normalized to a 1-week scale)
        urgency_clock = float(np.clip(self.bars_since_last_trade / 480.0, 0.0, 1.0))

        obs[-3] = equity_ratio
        obs[-2] = drawdown
        obs[-1] = urgency_clock

        return np.clip(obs, -10.0, 10.0)

    def step(self, action):
        direction_val, size_val, tp_val, sl_val = (
            action[0],
            action[1],
            action[2],
            action[3],
        )

        direction = 0
        if direction_val > 0.33:
            direction = 1
        elif direction_val < -0.33:
            direction = 2

        simulated_pnl = 0.0
        tp_mult_used = 0.0
        sl_mult_used = 0.0
        frequency_penalty = 0.0
        rr_penalty = 0.0

        self.bars_since_last_trade += 1

        if direction != 0:
            # --- CONSTRAINT 1: OVERTRADING PENALTY (Max 1/day) ---
            if self.bars_since_last_trade < self.min_bars_between_trades:
                direction = 0  # Block execution
                frequency_penalty = -5.0  # Massive catastrophic penalty for trying to spam trades
            else:
                # Valid Execution Sequence
                self.bars_since_last_trade = 0
                risk_pct = ((size_val + 1.0) / 2.0) * 0.05
                sl_mult_used = ((sl_val + 1.0) / 2.0) * 1.0 + 0.5
                tp_mult_used = sl_mult_used * (((tp_val + 1.0) / 2.0) * 2.0 + 1.0)
                amount_at_risk = self.balance * risk_pct
                
                # --- NEW CONSTRAINT: ASYMMETRIC R:R INTENT PENALTY ---
                theoretical_rr = tp_mult_used / sl_mult_used if sl_mult_used > 0 else 0
                if theoretical_rr < 2.0:
                    # Exponential decay penalty for low R:R intents (peaks at -8.0 for 0 R:R)
                    rr_penalty = -2.0 * (2.0 - theoretical_rr)**2 

                prob_win = (
                    self.df.loc[self.current_step, "prob_long"]
                    if direction == 1
                    else self.df.loc[self.current_step, "prob_short"]
                )

                if np.random.rand() < prob_win:
                    simulated_pnl = amount_at_risk * (tp_mult_used / sl_mult_used)
                else:
                    simulated_pnl = -amount_at_risk

                simulated_pnl -= self.friction_cost
        else:
            # --- CONSTRAINT 2: UNDERTRADING PENALTY (Urgency Decay) ---
            if self.bars_since_last_trade > self.urgency_threshold:
                # Creeping penalty encourages the agent to find a valid setup
                frequency_penalty = -0.05 * (
                    (self.bars_since_last_trade - self.urgency_threshold) / 100.0
                )

        self.balance += simulated_pnl

        peak_equity = max(self.initial_balance, self.peak_balance)
        self.peak_balance = max(peak_equity, self.balance)
        drawdown = (self.peak_balance - self.balance) / self.peak_balance

        # --- UPDATED REWARD TOPOGRAPHY ---
        # Reward is now actively heavily suppressed by poor R:R selection (rr_penalty)
        raw_reward = simulated_pnl - (drawdown * self.initial_balance * 0.1)
        reward = (raw_reward / (self.initial_balance * 0.01)) + frequency_penalty + rr_penalty

        reward = float(np.clip(reward, -10.0, 10.0))

        self.current_step += 1
        terminated = self.balance < (self.initial_balance * 0.1)
        truncated = self.current_step >= len(self.df) - 1

        info = {
            "balance": self.balance,
            "drawdown": drawdown,
            "tp_mult_used": tp_mult_used,
            "sl_mult_used": sl_mult_used,
            "rr_penalty": rr_penalty
        }

        return self._get_obs(), reward, terminated, truncated, info
