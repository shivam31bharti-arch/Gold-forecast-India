"""
train/splitter.py — Walk-forward TimeSeriesSplit for gold forecasting.

CRITICAL: No data leakage in time-series.
Training window is always 100% in the past relative to test window.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from typing import Generator, Tuple


def walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    min_train_size: int = 500,
    test_size: int = 60,
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Generator yielding (train_df, test_df) splits in chronological order.
    Each test window is strictly after training window — no leakage possible.

    Args:
        df: sorted DataFrame with DatetimeIndex
        n_splits: number of folds
        min_train_size: minimum training samples (default ~2yr daily)
        test_size: test window size in trading days
    """
    n = len(df)
    total_test = n_splits * test_size
    assert n > min_train_size + total_test, "Not enough data for requested splits."

    for i in range(n_splits):
        test_end = n - (n_splits - i - 1) * test_size
        test_start = test_end - test_size
        train_end = test_start

        actual_train_start = max(0, train_end - min_train_size * 2)  # rolling 2yr window
        train_df = df.iloc[actual_train_start:train_end]
        test_df  = df.iloc[test_start:test_end]

        yield train_df, test_df


def final_train_test_split(df: pd.DataFrame, test_fraction: float = 0.15):
    """
    Single hold-out split: last test_fraction for testing.
    Used for final model evaluation before deployment.
    """
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]
