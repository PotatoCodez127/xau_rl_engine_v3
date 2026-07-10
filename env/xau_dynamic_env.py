import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class XAUDynamicEnv(gym.Env):
    """
    XAUUSD Master-Slave Execution Environment (V3.2).
    Phase A (Oracle) strictly dictates direction. Phase B (SAC Manager) exclusively dictates sizing.
    Refactored: AOT C-contiguous memory allocation to eliminate Pandas CPU bottleneck.
    """
    def __init__(self, df: pd.DataFrame, initial_balance=10000.0):
        super(XAUDynamicEnv, self).__init__()
        
        # Drop index to ensure clean integer row mapping downstream
        self.df = df.reset_index(drop=True) 
        self.initial_balance = initial_balance
        self.friction_cost = 10.0

        # Structural Gating Constraints
        self.min_bars_between_trades = 4  # 1-hour anti-cluster cooldown
        self.max_trades_per_day = 1       # Prop-firm frequency survival constraint
        
        # --- ACTION SPACE ASYMMETRY BINDING ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Feature Pipeline
        exclude_cols = ["target", "time", "datetime", "date", "index"]
        self.feature_cols = [
            c for c in self.df.columns if c not in exclude_cols and not c.startswith("env_")
        ]

        # Observation Space
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(len(self.feature_cols) + 3,), dtype=np.float32
        )

        # --- AOT MEMORY ALLOCATION (Zero-Overhead NumPy Arrays) ---
        # Matrix for rapid _get_obs() slicing
        self.features_mat = np.ascontiguousarray(self.df[self.feature_cols].values, dtype=np.float32)
        
        # Fast-access 1D vectors for step() logic
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

    def reset(self, seed=None, options=None):
        """Resets the environment for the next WFA training epoch."""
        super().reset(seed=seed)
        self.current_step = 30
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.bars_since_last_trade = 0
        self.trades_today = 0
        self.current_day = self.dates_arr[self.current_step]
        return self._get_obs(), {}

    def _get_obs(self):
        """Builds the 1D Tensor state observation for the SAC Agent."""
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Zero-overhead C-level array slicing
        X = self.features_mat[self.current_step]
        obs[: len(X)] = X

        # Account Health Vectors
        obs[-3] = float(np.clip(self.balance / self.initial_balance, 0.0, 10.0))
        obs[-2] = float(np.clip((self.peak_balance - self.balance) / self.peak_balance, 0.0, 1.0))
        obs[-1] = float(np.clip(self.bars_since_last_trade / 480.0, 0.0, 1.0))
        
        return np.clip(obs, -10.0, 10.0)

    def _scale_action(self, action: np.ndarray) -> tuple[float, float]:
        """
        Maps the [-1, 1] SAC output to strictly bounded asymmetric risk profiles.
        This prevents Risk Inversion (e.g., risking $100 to make $45).
        """
        # SL Multiplier: 0.5x to 2.0x ATR
        sl_mult = 0.5 + ((action[0] + 1.0) * (2.0 - 0.5)) / 2.0
        
        # TP Multiplier: 1.0x to 3.0x of the chosen SL (ensuring mathematical positive expectancy)
        tp_mult = 1.0 + ((action[1] + 1.0) * (3.0 - 1.0)) / 2.0
        
        return sl_mult, tp_mult

    def _evaluate_master_slave_trigger(self, prob_long: float, prob_short: float, prob_hold: float, h4_trend: float) -> int:
        """
        Phase A Directional Logic. Evaluates Relative Conviction and the absolute H4 Macro Gate.
        Returns: 1 (Long), 2 (Short), 0 (Hold)
        """
        EXECUTION_THRESHOLD = 0.35
        
        # Relative Conviction + Macro Gating (H4 Trend Alignment)
        if prob_long > EXECUTION_THRESHOLD and prob_long > prob_hold and h4_trend > 0:
            return 1
        elif prob_short > EXECUTION_THRESHOLD and prob_short > prob_hold and h4_trend < 0:
            return 2
            
        return 0

    def step(self, action):
        step_date = self.dates_arr[self.current_step]
        
        # Automated CI Runner Temporal Synchronization (UTC)
        if step_date > self.current_day:
            self.current_day = step_date
            self.trades_today = 0

        # Unpack and bound the action space
        sl_mult_used, tp_mult_ratio = self._scale_action(action)
        tp_mult_used = sl_mult_used * tp_mult_ratio 

        # Oracle Master-Slave Logic Extraction (O(1) Array Indexing)
        prob_long = self.prob_long_arr[self.current_step]
        prob_short = self.prob_short_arr[self.current_step]
        prob_hold = self.prob_hold_arr[self.current_step]
        h4_trend = self.h4_trend_arr[self.current_step]
        
        direction = self._evaluate_master_slave_trigger(prob_long, prob_short, prob_hold, h4_trend)

        # Operational Constraints (Frequency Gating)
        if self.bars_since_last_trade < self.min_bars_between_trades or self.trades_today >= self.max_trades_per_day:
            direction = 0

        simulated_pnl = 0.0
        self.bars_since_last_trade += 1
        
        # --- EPISODIC REWARD ENGINEERING ---
        reward = 0.0  

        if direction != 0:
            self.bars_since_last_trade = 0
            self.trades_today += 1
            
            # Dynamic Compounding: Fixed 1.5% Risk Protocol
            amount_at_risk = self.balance * 0.015
            prob_win = prob_long if direction == 1 else prob_short

            if np.random.rand() < prob_win:
                simulated_pnl = amount_at_risk * (tp_mult_used / sl_mult_used)
            else:
                simulated_pnl = -amount_at_risk

            simulated_pnl -= self.friction_cost
            self.balance += simulated_pnl
            self.peak_balance = max(max(self.initial_balance, self.peak_balance), self.balance)
            
            # --- TERMINAL CALMAR REWARD PROXY ---
            drawdown = (self.peak_balance - self.balance) / self.peak_balance
            drawdown_penalty = drawdown * self.initial_balance * 0.25
            
            raw_reward = simulated_pnl - drawdown_penalty
            
            # Normalize to [-10, 10] range for SAC stability
            reward = float(np.clip((raw_reward / (self.initial_balance * 0.01)), -10.0, 10.0))

        # Check termination conditions (Account Blown)
        self.current_step += 1
        terminated = self.balance < (self.initial_balance * 0.1)
        truncated = self.current_step >= self.max_steps

        info = {
            "prob_long": prob_long,
            "prob_short": prob_short,
            "sl_mult_used": sl_mult_used,
            "tp_mult_used": tp_mult_used,
        }

        return self._get_obs(), reward, terminated, truncated, info