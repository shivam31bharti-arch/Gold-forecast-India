"""
regime/garch_model.py — GARCH(1,1) conditional variance as a feature.

GARCH is used as a FEATURE (not a regime classifier).
It captures volatility clustering — periods of high variance follow each other.
This is separately validated on MCX gold in Indian commodity research (SSRN).

Usage:
  python regime/garch_model.py
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from arch import arch_model

from utils.config_loader import load_config
from utils.logger import get_logger

log = get_logger("garch_model")


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1):
    """
    Fit GARCH(p,q) on daily log return series.
    Returns the fitted result object.
    """
    # Scale returns to % (arch library prefers percentage returns)
    scaled = returns * 100
    am = arch_model(scaled, vol="Garch", p=p, q=q, dist="Normal")
    result = am.fit(disp="off", show_warning=False)
    log.info(
        f"GARCH({p},{q}) fitted | AIC={result.aic:.2f} | BIC={result.bic:.2f} | "
        f"alpha={result.params.get('alpha[1]', float('nan')):.4f} | "
        f"beta={result.params.get('beta[1]', float('nan')):.4f}"
    )
    return result


def extract_conditional_variance(result, returns: pd.Series) -> pd.Series:
    """
    Extract in-sample conditional variance σ²_t from fitted GARCH model.
    Scaled back to original return units (divides by 100²).
    """
    cond_var = result.conditional_volatility ** 2 / 10000  # back to decimals²
    cond_var.index = returns.index
    cond_var.name = "garch_cond_var"
    return cond_var


def add_garch_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Fit GARCH on MCX log returns, add conditional variance and derived features.
    """
    gcfg = cfg["garch"]
    returns = df["mcx_approx_logret"].dropna()

    if len(returns) < 100:
        log.warning("Too few observations for GARCH. Adding zero-filled placeholder.")
        df["garch_cond_var"] = 0.0
        df["garch_high_vol_flag"] = 0
        return df

    try:
        result = fit_garch(returns, p=gcfg["p"], q=gcfg["q"])
        cond_var = extract_conditional_variance(result, returns)

        df["garch_cond_var"] = cond_var
        df["garch_cond_var"] = df["garch_cond_var"].ffill().fillna(0)

        # Derived binary flag: HIGH if cond_var > 90th percentile
        threshold = df["garch_cond_var"].quantile(0.90)
        df["garch_high_vol_flag"] = (df["garch_cond_var"] > threshold).astype(int)

        # Log-transform variance (less skewed)
        df["garch_log_var"] = np.log1p(df["garch_cond_var"])

        # Save fitted model
        artifacts_path = cfg["paths"]["model_artifacts"]
        os.makedirs(artifacts_path, exist_ok=True)
        joblib.dump(result, os.path.join(artifacts_path, "garch_result.pkl"))
        log.info("GARCH result saved.")

    except Exception as e:
        log.error(f"GARCH fitting failed: {e}. Adding zeroed placeholders.")
        df["garch_cond_var"] = 0.0
        df["garch_high_vol_flag"] = 0
        df["garch_log_var"] = 0.0

    return df


if __name__ == "__main__":
    cfg = load_config()
    fpath = os.path.join(cfg["paths"]["processed_data"], "features_daily.parquet")
    df = pd.read_parquet(fpath)
    df = add_garch_features(df, cfg)
    df.to_parquet(fpath)  # save back in place

    print(df[["mcx_approx_logret", "garch_cond_var", "garch_high_vol_flag",
               "garch_log_var"]].tail(20))
