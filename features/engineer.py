"""
features/engineer.py — Core feature engineering for the gold forecasting model.

Generates:
  - Lag features (1, 2, 3, 7, 14 days)
  - Rolling mean / std / min / max
  - Volatility cluster flag
  - Log return stationarity
  - Seasonal features (delegates to seasonal.py)
  - Target columns for each horizon

Usage:
  python features/engineer.py
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from features.seasonal import add_seasonal_features
from utils.config_loader import load_config
from utils.logger import get_logger

log = get_logger("feature_engineer")


def add_lag_features(df: pd.DataFrame, col: str, lags: list) -> pd.DataFrame:
    """Add lagged values of a column."""
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str, windows: list) -> pd.DataFrame:
    """Add rolling mean, std, min, max for given windows."""
    df = df.copy()
    for w in windows:
        df[f"{col}_rmean{w}"] = df[col].rolling(w, min_periods=max(1, w//2)).mean()
        df[f"{col}_rstd{w}"]  = df[col].rolling(w, min_periods=max(2, w//2)).std()
        df[f"{col}_rmin{w}"]  = df[col].rolling(w, min_periods=max(1, w//2)).min()
        df[f"{col}_rmax{w}"]  = df[col].rolling(w, min_periods=max(1, w//2)).max()
    return df


def add_volatility_cluster_flag(df: pd.DataFrame, logret_col: str,
                                 threshold_pct: int = 75) -> pd.DataFrame:
    """
    Binary flag: 1 if |logret| > threshold_percentile of rolling 30d volatility.
    Captures GARCH-style volatility clustering.
    """
    df = df.copy()
    rolling_vol = df[logret_col].rolling(30, min_periods=10).std()
    threshold = rolling_vol.quantile(threshold_pct / 100)
    df["vol_cluster_flag"] = (rolling_vol > threshold).astype(int)
    return df


def add_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-asset interaction features."""
    df = df.copy()
    # USD/INR momentum (key driver of Indian gold premium)
    if "usdinr_logret" in df.columns:
        df["usdinr_7d_momentum"] = df["usdinr_logret"].rolling(7).sum()

    # Gold-crude correlation proxy (crude affects INR which affects gold)
    if "crude_usd_logret" in df.columns and "gold_usd_logret" in df.columns:
        df["gold_crude_spread_logret"] = df["gold_usd_logret"] - df["crude_usd_logret"]

    # DXY-Gold inverse relationship proxy
    if "dxy_logret" in df.columns and "gold_usd_logret" in df.columns:
        df["gold_dxy_spread"] = df["gold_usd_logret"] - df["dxy_logret"]

    return df


def add_target_columns(df: pd.DataFrame, horizons: list,
                        target_col: str = "mcx_approx_logret") -> pd.DataFrame:
    """
    Build forward-looking target for each horizon.
    target_h = log return h days ahead = log(price_{t+h} / price_{t})
    Uses shift(-h) — NEVER use future data as a feature (no leakage).
    """
    df = df.copy()
    for h in horizons:
        # Rolling h-day log return forward
        df[f"target_{h}d"] = df[target_col].shift(-h).rolling(h).sum().shift(-(h-1))
        # Also store direction (1=up, 0=down)
        df[f"target_{h}d_dir"] = (df[f"target_{h}d"] > 0).astype(int)
    return df


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    fcfg = cfg["features"]
    lags = fcfg["lags"]
    windows = fcfg["rolling_windows"]
    vol_thresh = fcfg["vol_cluster_threshold_percentile"]
    horizons = cfg["training"]["horizons"]

    primary_col = "mcx_approx_logret"          # primary return series
    lag_cols = ["gold_usd_logret", "usdinr_logret", "dxy_logret",
                "crude_usd_logret", primary_col]

    for col in lag_cols:
        if col in df.columns:
            df = add_lag_features(df, col, lags)
            df = add_rolling_features(df, col, windows)

    df = add_volatility_cluster_flag(df, primary_col, vol_thresh)
    df = add_cross_features(df)
    df = add_seasonal_features(df)
    df = add_target_columns(df, horizons, primary_col)

    # Drop rows at the head that have NaN from lagging
    df = df.dropna(subset=[f"target_{horizons[-1]}d"])  # drop where longest target missing
    log.info(f"Feature matrix built: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def save_features(df: pd.DataFrame, cfg: dict) -> str:
    path = cfg["paths"]["processed_data"]
    os.makedirs(path, exist_ok=True)
    fpath = os.path.join(path, "features_daily.parquet")
    df.to_parquet(fpath)
    log.info(f"Feature matrix saved to {fpath}")
    return fpath


def get_feature_columns(df: pd.DataFrame, horizons: list) -> list:
    """Return feature column names (exclude targets and metadata)."""
    exclude = {"outlier_flag", "month", "day_of_week", "day_of_year", "quarter"}
    target_cols = set()
    for h in horizons:
        target_cols.add(f"target_{h}d")
        target_cols.add(f"target_{h}d_dir")

    # Also exclude raw price columns
    raw_price = {"gold_usd", "usdinr", "dxy", "crude_usd", "mcx_approx",
                 "CPIAUCSL", "DGS10"}

    return [c for c in df.columns
            if c not in exclude and c not in target_cols and c not in raw_price
            and not c.endswith(("_logret",)) or c.endswith(("lag1", "rmean7"))]


if __name__ == "__main__":
    cfg = load_config()
    processed_path = os.path.join(cfg["paths"]["processed_data"], "processed_daily.parquet")
    df = pd.read_parquet(processed_path)
    df = build_features(df, cfg)
    save_features(df, cfg)
    print(df.shape)
    print(df.columns.tolist())
