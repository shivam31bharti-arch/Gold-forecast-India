"""
data/preprocess.py — Cleaning, log return computation, and MCX price approximation.

MCX price approximation:
  MCX (₹/10g) ≈ USD_gold_per_oz * USDINR * import_factor / 31.1035 * 10
  (31.1035 g/troy oz)

Usage:
  python data/preprocess.py
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from utils.config_loader import load_config
from utils.logger import get_logger

log = get_logger("preprocess")


def compute_mcx_approx(df: pd.DataFrame, import_factor: float = 1.125) -> pd.DataFrame:
    """
    Approximate MCX gold price in ₹/10g from USD gold + USD/INR.
    Formula: (USD_oz * USDINR * import_factor / 31.1035) * 10
    """
    df = df.copy()
    if "gold_usd" in df.columns and "usdinr" in df.columns:
        df["mcx_approx"] = (
            df["gold_usd"] * df["usdinr"] * import_factor / 31.1035 * 10
        )
        log.info("MCX approximation column added.")
    else:
        log.warning("Missing gold_usd or usdinr — cannot compute mcx_approx.")
    return df


def compute_log_returns(df: pd.DataFrame, price_cols: list) -> pd.DataFrame:
    """Compute daily log returns for each price column."""
    df = df.copy()
    for col in price_cols:
        if col in df.columns:
            df[f"{col}_logret"] = np.log(df[col] / df[col].shift(1))
    log.info(f"Log returns computed for: {price_cols}")
    return df


def handle_missing(df: pd.DataFrame, max_consecutive_gap: int = 5) -> pd.DataFrame:
    """
    Forward-fill short gaps (≤ max_consecutive_gap days).
    Drop rows where core price columns still NaN after fill.
    """
    df = df.copy()
    df = df.ffill(limit=max_consecutive_gap)
    before = len(df)
    df = df.dropna(subset=["gold_usd"])
    after = len(df)
    if before != after:
        log.info(f"Dropped {before - after} rows with missing gold price after forward-fill.")
    return df


def remove_extreme_outliers(df: pd.DataFrame, col: str = "gold_usd_logret",
                             z_thresh: float = 5.0) -> pd.DataFrame:
    """Flag (not remove) extreme log return outliers beyond z_thresh standard deviations."""
    df = df.copy()
    if col not in df.columns:
        return df
    mean = df[col].mean()
    std = df[col].std()
    df["outlier_flag"] = (np.abs(df[col] - mean) > z_thresh * std).astype(int)
    n_outliers = df["outlier_flag"].sum()
    if n_outliers:
        log.warning(f"{n_outliers} extreme outlier(s) flagged in {col} (z > {z_thresh})")
    return df


def preprocess(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    import_factor = cfg["conversion"]["import_factor"]
    price_cols = ["gold_usd", "usdinr", "dxy", "crude_usd", "mcx_approx"]

    df = handle_missing(df)
    df = compute_mcx_approx(df, import_factor)
    df = compute_log_returns(df, price_cols)
    df = remove_extreme_outliers(df, col="gold_usd_logret")
    df = df.dropna(subset=["gold_usd_logret"])

    log.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df


def save_processed(df: pd.DataFrame, cfg: dict) -> str:
    path = cfg["paths"]["processed_data"]
    os.makedirs(path, exist_ok=True)
    fpath = os.path.join(path, "processed_daily.parquet")
    df.to_parquet(fpath)
    log.info(f"Processed data saved to {fpath}")
    return fpath


if __name__ == "__main__":
    cfg = load_config()
    raw_path = os.path.join(cfg["paths"]["raw_data"], "raw_daily.parquet")
    df = pd.read_parquet(raw_path)
    df = preprocess(df, cfg)
    save_processed(df, cfg)
    print(df[["gold_usd", "usdinr", "mcx_approx", "mcx_approx_logret"]].tail(10))
