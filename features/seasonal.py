"""
features/seasonal.py — Indian gold market seasonal calendar.

Key demand spikes in Indian gold market:
  - Wedding season: Nov-Feb (peak) and Apr-May
  - Akshaya Tritiya: late Apr / early May (auspicious gold buying day)
  - Diwali: Oct-Nov (Dhanteras — major gold buying day)
  - Gudi Padwa / Ugadi: Mar-Apr
  - Monsoon slowdown: Jun-Aug (demand dip)
"""
import pandas as pd
import numpy as np


# Fixed-window seasonal flags (month-based approximation)
WEDDING_SEASON_MONTHS = {11, 12, 1, 2}       # Nov–Feb (peak)
WEDDING_SEASON_MINOR_MONTHS = {4, 5}          # Apr–May
MONSOON_MONTHS = {6, 7, 8}                    # Jun–Aug (demand dip)
FESTIVE_MONTHS = {10, 11}                     # Oct–Nov (Diwali / Dhanteras)

# Akshaya Tritiya approx: 3rd Tithi of Vaishakha Shukla Paksha → usually late Apr/May
AKSHAYA_TRITIYA_APPROX_MONTH = 5             # May (approximate)
AKSHAYA_TRITIYA_APPROX_DAY_RANGE = (1, 15)   # First half of May


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Indian seasonal demand features to a daily-indexed DataFrame.
    All features are binary (0/1) or cyclical sine/cosine encodings.
    """
    df = df.copy()
    idx = df.index

    # Basic calendar
    df["month"] = idx.month
    df["day_of_week"] = idx.dayofweek    # 0=Mon, 6=Sun
    df["day_of_year"] = idx.dayofyear
    df["quarter"] = idx.quarter

    # Cyclical encoding of month (preserves circular nature)
    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)

    # Cyclical encoding of day-of-week
    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

    # Binary seasonal flags
    df["wedding_season_peak"] = idx.month.isin(WEDDING_SEASON_MONTHS).astype(int)
    df["wedding_season_minor"] = idx.month.isin(WEDDING_SEASON_MINOR_MONTHS).astype(int)
    df["monsoon_slowdown"] = idx.month.isin(MONSOON_MONTHS).astype(int)
    df["festive_season"] = idx.month.isin(FESTIVE_MONTHS).astype(int)

    # Akshaya Tritiya window flag (approx May 1–15)
    lo, hi = AKSHAYA_TRITIYA_APPROX_DAY_RANGE
    df["akshaya_tritiya_window"] = (
        (idx.month == AKSHAYA_TRITIYA_APPROX_MONTH) &
        (idx.day >= lo) & (idx.day <= hi)
    ).astype(int)

    # Combined demand pressure index (unsigned, additive proxy)
    df["demand_pressure"] = (
        df["wedding_season_peak"] * 2 +
        df["wedding_season_minor"] * 1 +
        df["festive_season"] * 1.5 +
        df["akshaya_tritiya_window"] * 1 -
        df["monsoon_slowdown"] * 1
    )

    return df
