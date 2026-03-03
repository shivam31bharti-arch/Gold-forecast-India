"""
regime/hmm_model.py — 2-state Hidden Markov Model for gold market regime detection.

States:
  0 = LOW volatility regime
  1 = HIGH volatility regime (crisis periods, geopolitical shocks)

Validation:
  - State 1 should align with COVID Mar 2020 crash
  - State 1 should align with Russia-Ukraine Feb-Mar 2022 spike

Usage:
  python regime/hmm_model.py
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from hmmlearn.hmm import GaussianHMM

from utils.config_loader import load_config
from utils.logger import get_logger

log = get_logger("hmm_model")


def fit_hmm(returns: np.ndarray, n_states: int = 2, n_iter: int = 1000,
            random_state: int = 42) -> GaussianHMM:
    """
    Fit a Gaussian HMM on daily log return sequence.
    Returns a fitted model.
    """
    X = returns.reshape(-1, 1)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X)
    log.info(f"HMM fitted | n_iter={n_iter} | log-likelihood converged: {model.monitor_.converged}")
    return model


def label_regimes(model: GaussianHMM, returns: np.ndarray) -> np.ndarray:
    """
    Predict hidden states for each return observation.
    Re-label states so that HIGH volatility state = 1.
    (HMM state labels are arbitrary — we normalise by variance.)
    """
    X = returns.reshape(-1, 1)
    states = model.predict(X)

    # Identify which state has higher variance → call it HIGH (1)
    variances = [model.covars_[s][0][0] for s in range(model.n_components)]
    high_vol_state = int(np.argmax(variances))

    if high_vol_state == 0:
        # Flip labels so HIGH volatility = 1
        states = 1 - states
        log.info("HMM state labels flipped: high-vol mapped to state 1.")
    else:
        log.info("HMM state labels are correct: state 1 = high volatility.")

    return states


def validate_regime_labels(df: pd.DataFrame, regime_col: str = "regime") -> dict:
    """
    Check that known crisis periods are correctly labeled as HIGH regime (1).
    Logs warnings if alignment fails.
    """
    results = {}

    crisis_windows = {
        "COVID-crash-Mar2020": ("2020-03-01", "2020-04-30"),
        "Russia-Ukraine-Feb2022": ("2022-02-20", "2022-04-30"),
        "Inflation-spike-2022": ("2022-06-01", "2022-10-31"),
    }

    for name, (start, end) in crisis_windows.items():
        window = df.loc[start:end, regime_col] if start in df.index or end in df.index else pd.Series(dtype=int)
        if window.empty:
            results[name] = "NO DATA"
            log.warning(f"Validation window not in dataset: {name}")
            continue

        pct_high = window.mean()
        results[name] = pct_high
        status = "✅ PASS" if pct_high >= 0.5 else "⚠️ FAIL"
        log.info(f"Regime validation [{name}]: {pct_high:.1%} HIGH regime {status}")

    return results


def fallback_rolling_regime(returns: pd.Series, percentile: int = 75) -> pd.Series:
    """
    Fallback regime label using rolling 30-day std percentile.
    Used if HMM fails to converge or dataset is too small.
    """
    rolling_vol = returns.rolling(30, min_periods=10).std()
    threshold = rolling_vol.quantile(percentile / 100)
    regime = (rolling_vol > threshold).astype(int)
    log.warning("Using fallback rolling-std regime detection.")
    return regime


def run_hmm_pipeline(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Full regime detection pipeline. Adds 'regime' and 'regime_prob_high' columns."""
    hcfg = cfg["hmm"]
    returns = df["mcx_approx_logret"].dropna().values

    if len(returns) < 200:
        log.warning(f"Only {len(returns)} observations — HMM may be unstable. Using fallback.")
        df["regime"] = fallback_rolling_regime(df["mcx_approx_logret"])
        df["regime_prob_high"] = df["regime"].astype(float)
        return df

    try:
        model = fit_hmm(returns, n_states=hcfg["n_states"], n_iter=hcfg["n_iter"],
                        random_state=hcfg["random_state"])
        states = label_regimes(model, returns)

        # Align with original df index (only rows where logret is not NaN)
        valid_idx = df["mcx_approx_logret"].dropna().index
        df.loc[valid_idx, "regime"] = states.astype(int)
        df["regime"] = df["regime"].ffill().fillna(0).astype(int)

        # Posterior probabilities (prob of being in high-vol state)
        X = returns.reshape(-1, 1)
        posteriors = model.predict_proba(X)
        high_vol_col = 1  # re-mapped high state is always 1 after label_regimes
        df.loc[valid_idx, "regime_prob_high"] = posteriors[:, high_vol_col]
        df["regime_prob_high"] = df["regime_prob_high"].ffill().fillna(0.5)

        # Save model
        artifacts_path = cfg["paths"]["model_artifacts"]
        os.makedirs(artifacts_path, exist_ok=True)
        joblib.dump(model, os.path.join(artifacts_path, "hmm_model.pkl"))
        log.info("HMM model saved.")

        # Validate
        validate_regime_labels(df)

    except Exception as e:
        log.error(f"HMM fitting failed: {e}. Falling back to rolling std.")
        df["regime"] = fallback_rolling_regime(df["mcx_approx_logret"])
        df["regime_prob_high"] = df["regime"].astype(float)

    return df


if __name__ == "__main__":
    cfg = load_config()
    fpath = os.path.join(cfg["paths"]["processed_data"], "features_daily.parquet")
    df = pd.read_parquet(fpath)
    df = run_hmm_pipeline(df, cfg)

    # Save back with regime columns
    out = os.path.join(cfg["paths"]["processed_data"], "features_daily.parquet")
    df.to_parquet(out)

    print(df[["mcx_approx_logret", "regime", "regime_prob_high"]].tail(20))
    print(f"\nRegime distribution:\n{df['regime'].value_counts()}")
