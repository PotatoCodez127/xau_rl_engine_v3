import time
import pandas as pd
import numpy as np
from collections import deque

class M1DataFeed:
    """Mimics the MetaTrader 5 live data feed."""
    def __init__(self, raw_m1_csv_path):
        self.filepath = raw_m1_csv_path
        print(f"Initializing M1 Data Feed from: {raw_m1_csv_path}")

    def stream(self):
        """Yields one M1 bar at a time to simulate live market ticks."""
        # Using chunksize to prevent loading massive M1 files completely into memory
        chunk_iter = pd.read_csv(self.filepath, sep="\t", chunksize=10000)
        for chunk in chunk_iter:
            if len(chunk.columns) == 1:
                # Fallback for comma separated
                chunk = pd.read_csv(self.filepath, sep=",", chunksize=10000)
                break
                
        # Re-initialize properly based on separator
        separator = "\t" if len(pd.read_csv(self.filepath, nrows=1, sep="\t").columns) > 1 else ","
        chunk_iter = pd.read_csv(self.filepath, sep=separator, chunksize=10000)
        
        for chunk in chunk_iter:
            chunk.columns = [c.strip("<>").lower() for c in chunk.columns]
            chunk["datetime"] = pd.to_datetime(chunk["date"] + " " + chunk["time"], format="%Y.%m.%d %H:%M:%S")
            chunk.set_index("datetime", inplace=True)
            chunk.drop(columns=["date", "time", "tickvol", "vol", "spread"], inplace=True, errors="ignore")
            
            for index, row in chunk.iterrows():
                yield index, row


class StreamingFeatureEngine:
    """Stateful feature builder that mimics live EA calculation."""
    def __init__(self, window_size=1000):
        # We need at least 800 periods of 15m history for the H4 Trend EMA
        self.history_limit = window_size 
        self.m1_buffer = []
        self.m15_history = pd.DataFrame()
        self.is_warmed_up = False

    def process_m1_tick(self, timestamp, m1_row):
        """Absorbs an M1 tick. Returns Features if a 15m bar just closed, else None."""
        self.m1_buffer.append(m1_row)
        
        # Check if the 15m candle has closed (00, 15, 30, 45)
        # Note: In MT5, a bar closes when the *first tick* of the next bar arrives.
        if timestamp.minute % 15 == 0 and len(self.m1_buffer) > 0:
            features = self._close_15m_candle(timestamp)
            self.m1_buffer = [] # Reset M1 buffer for next 15m candle
            return features
        return None

    def _close_15m_candle(self, timestamp):
        """Synthesizes the 15m candle and calculates the latest feature row."""
        if not self.m1_buffer:
            return None
            
        m1_df = pd.DataFrame(self.m1_buffer)
        
        # 1. Synthesize the 15m Bar
        new_15m = pd.DataFrame({
            "open": [m1_df["open"].iloc[0]],
            "high": [m1_df["high"].max()],
            "low": [m1_df["low"].min()],
            "close": [m1_df["close"].iloc[-1]]
        }, index=[timestamp])

        # 2. Append to rolling history
        self.m15_history = pd.concat([self.m15_history, new_15m])
        
        # Truncate history to save memory (Keep last 1000 15m bars)
        if len(self.m15_history) > self.history_limit:
            self.m15_history = self.m15_history.iloc[-self.history_limit:]
            self.is_warmed_up = True

        # 3. Calculate Features (Only if warmed up to prevent NaN errors)
        if self.is_warmed_up:
            return self._calculate_current_features()
        return None

    def _calculate_current_features(self):
        """Runs the exact V3 math from build_features.py on the rolling window."""
        df = self.m15_history.copy()
        
        # ATR (14)
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        df["env_atr"] = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1).rolling(14).mean()

        # EMAs & Macro Context
        ema_50h = df["close"].ewm(span=200, adjust=False).mean()
        ema_200h = df["close"].ewm(span=800, adjust=False).mean()
        df["h4_trend"] = (ema_50h - ema_200h) / ema_200h

        # Volatility Regime
        h1_high = df["high"].rolling(window=4).max()
        h1_low = df["low"].rolling(window=4).min()
        h1_range = h1_high - h1_low
        df["h1_vol_regime"] = h1_range.rolling(window=96).rank(pct=True)
        
        # Fractional Differentiation (Window 50)
        # Note: In a production EA, we will optimize this to only calculate the dot product 
        # of the last 50 bars, not the whole dataframe, to save microseconds.
        weights = self._get_fractional_weights(0.45, 50)
        last_50_close = df["close"].iloc[-50:].values
        df.loc[df.index[-1], "close_frac_diff"] = np.dot(last_50_close, weights)[0]

        # Return ONLY the latest row (the live present) as a dictionary for the Simulator
        return df.iloc[-1].to_dict()

    def _get_fractional_weights(self, d: float, size: int) -> np.ndarray:
        w = [1.0]
        for k in range(1, size):
            w_ = -w[-1] / k * (d - k + 1)
            w.append(w_)
        return np.array(w[::-1]).reshape(-1, 1)