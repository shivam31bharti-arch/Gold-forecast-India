"""
data/ingest.py — Pull gold price, macro, and forex data from free sources.

Sources:
  - yfinance: GC=F (gold futures), USDINR=X, DX-Y.NYB (DXY), CL=F (crude)
  - fredapi: DGS10 (10y yield), CPIAUCSL (CPI)

Usage:
  python data/ingest.py
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from utils.config_loader import load_config
from utils.logger import get_logger

load_dotenv()
log = get_logger("ingest")


def fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance, return daily Close series."""
    log.info(f"Fetching {ticker} from {start} to {end}")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        log.warning(f"No data returned for {ticker}")
        return pd.DataFrame()
    df = df[["Close"]].rename(columns={"Close": ticker})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def fetch_fred(series_id: str, start: str, end: str) -> pd.Series:
    """Download a FRED series. Requires FRED_API_KEY in environment."""
    try:
        from fredapi import Fred
    except ImportError:
        log.error("fredapi not installed. Run: pip install fredapi")
        raise

    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        log.warning("FRED_API_KEY not set — FRED data will be skipped.")
        return pd.Series(dtype=float, name=series_id)

    fred = Fred(api_key=api_key)
    log.info(f"Fetching FRED series: {series_id}")
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = series_id
    return s


def ingest_all(cfg: dict) -> pd.DataFrame:
    """
    Pull all data sources and join on a common daily index.
    Returns a wide DataFrame with columns for each source.
    """
    dcfg = cfg["data"]
    start = dcfg["start_date"]
    end = pd.Timestamp.today().strftime("%Y-%m-%d") if dcfg["end_date"] == "auto" else dcfg["end_date"]

    # --- yfinance sources ---
    tickers = {
        "gold_usd":  dcfg["gold_ticker"],
        "usdinr":    dcfg["usdinr_ticker"],
        "dxy":       dcfg["dxy_ticker"],
        "crude_usd": dcfg["crude_ticker"],
    }

    frames = []
    for col_name, ticker in tickers.items():
        raw = fetch_yfinance(ticker, start, end)
        if not raw.empty:
            raw.columns = [col_name]
            frames.append(raw)

    # --- FRED sources ---
    for series_id in [dcfg["fred_10y_series"], dcfg["fred_cpi_series"]]:
        s = fetch_fred(series_id, start, end)
        if not s.empty:
            frames.append(s.to_frame())

    if not frames:
        raise RuntimeError("No data sources returned data. Check network / API keys.")

    # Outer join on date index, forward-fill macro series (weekly/monthly → daily)
    df = frames[0]
    for f in frames[1:]:
        df = df.join(f, how="outer")

    # Forward fill macro/weekly series; drop weekends with no price data
    df = df.sort_index().ffill()
    df = df.dropna(subset=["gold_usd"])  # require at minimum gold price

    log.info(f"Ingested data: {df.shape[0]} rows × {df.shape[1]} cols | {df.index[0]} → {df.index[-1]}")
    return df


def save_raw(df: pd.DataFrame, cfg: dict) -> str:
    """Save raw data to parquet."""
    path = cfg["paths"]["raw_data"]
    os.makedirs(path, exist_ok=True)
    fpath = os.path.join(path, "raw_daily.parquet")
    df.to_parquet(fpath)
    log.info(f"Raw data saved to {fpath}")
    return fpath


if __name__ == "__main__":
    cfg = load_config()
    df = ingest_all(cfg)
    save_raw(df, cfg)
    print(df.tail())
    print(f"\nColumns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
