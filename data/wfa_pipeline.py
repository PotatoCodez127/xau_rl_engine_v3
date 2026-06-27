import pandas as pd
from typing import List, Dict


class WalkForwardPipeline:
    def __init__(self, features_path: str, embargo_bars: int = 50):
        """
        embargo_bars: 50 bars on a 15m timeframe = 12.5 hours of strict embargo
        """
        self.features_path = features_path
        self.embargo_bars = embargo_bars
        self.master_df = None

    def load_data(self, holdout_fraction: float = 0.2) -> pd.DataFrame:
        self.master_df = pd.read_csv(self.features_path, index_col=0, parse_dates=True)
        if holdout_fraction > 0:
            train_boundary = int(len(self.master_df) * (1 - holdout_fraction))
            self.master_df = self.master_df.iloc[:train_boundary]

        return self.master_df

    def generate_splits(
        self, train_size: int, test_size: int, step_size: int
    ) -> List[Dict[str, pd.DataFrame]]:
        if self.master_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        splits = []
        total_len = len(self.master_df)

        for i in range(0, total_len - train_size - test_size, step_size):
            train_end = i + train_size
            test_start = train_end + self.embargo_bars
            test_end = test_start + test_size

            if test_end > total_len:
                break

            splits.append(
                {
                    "train": self.master_df.iloc[i:train_end],
                    "test": self.master_df.iloc[test_start:test_end],
                }
            )

        return splits
