"""
inference/predictor.py — Load correct regime model and run prediction.

At inference time:
  1. Detect current regime from latest returns
  2. Load XGBoost model for that regime × requested horizon
  3. Return predicted log return and ₹/10g change
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Optional

from utils.config_loader import load_config
from utils.logger import get_logger

log = get_logger("predictor")


def load_model(regime: int, horizon: int, artifacts_path: str):
    """Load a specific XGBoost model from disk."""
    model_path = os.path.join(artifacts_path, f"xgb_r{regime}_{horizon}d.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run trainer.py first.")
    return joblib.load(model_path)


def load_feature_cols(artifacts_path: str) -> list:
    """Load the feature column list saved during training."""
    path = os.path.join(artifacts_path, "feature_cols.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("feature_cols.pkl missing. Run trainer.py first.")
    return joblib.load(path)


def detect_current_regime(df: pd.DataFrame, artifacts_path: str,
                           fallback_pct: int = 75) -> int:
    """
    Load saved HMM model and predict regime for the most recent data window.
    Falls back to rolling-std threshold if HMM model not found.
    """
    hmm_path = os.path.join(artifacts_path, "hmm_model.pkl")
    if os.path.exists(hmm_path):
        try:
            hmm_model = joblib.load(hmm_path)
            recent_returns = df["mcx_approx_logret"].dropna().tail(60).values
            X = recent_returns.reshape(-1, 1)
            states = hmm_model.predict(X)
            current_state = int(states[-1])

            # Check if high-vol state needs re-mapping
            variances = [hmm_model.covars_[s][0][0] for s in range(hmm_model.n_components)]
            high_vol_state = int(np.argmax(variances))
            if high_vol_state == 0:
                current_state = 1 - current_state

            log.info(f"HMM regime detected: {current_state} ({'HIGH' if current_state else 'LOW'})")
            return current_state
        except Exception as e:
            log.warning(f"HMM inference failed: {e}. Using fallback.")

    # Fallback: rolling std
    rolling_vol = df["mcx_approx_logret"].rolling(30).std()
    threshold = rolling_vol.quantile(fallback_pct / 100)
    current_vol = rolling_vol.iloc[-1]
    regime = int(current_vol > threshold)
    log.info(f"Fallback regime: {regime} (rolling_vol={current_vol:.6f}, threshold={threshold:.6f})")
    return regime


def predict_all_horizons(
    df: pd.DataFrame,
    cfg: dict,
    horizons: Optional[list] = None
) -> Dict[int, dict]:
    """
    Run predictions for all horizons under the current regime.

    Returns dict: {horizon: {log_return, inr_change, direction_prob, regime}}
    """
    artifacts_path = cfg["paths"]["model_artifacts"]
    if horizons is None:
        horizons = cfg["training"]["horizons"]

    feature_cols = load_feature_cols(artifacts_path)
    regime = detect_current_regime(df, artifacts_path)

    # Get the latest row of features
    latest = df[feature_cols].dropna().tail(1)
    if latest.empty:
        raise ValueError("No complete feature row available for prediction.")

    prev_mcx = df["mcx_approx"].dropna().iloc[-1]
    results = {}

    for horizon in horizons:
        try:
            model = load_model(regime, horizon, artifacts_path)
            log_return_pred = float(model.predict(latest)[0])
            inr_change = prev_mcx * (np.exp(log_return_pred) - 1)
            direction_prob = float(np.clip((log_return_pred + 0.02) / 0.04, 0, 1))

            results[horizon] = {
                "horizon_days": horizon,
                "regime": regime,
                "regime_label": "HIGH" if regime else "LOW",
                "predicted_log_return": round(log_return_pred, 6),
                "predicted_inr_change": round(inr_change, 2),
                "predicted_inr_price": round(prev_mcx + inr_change, 2),
                "direction_prob": round(direction_prob, 4),
                "prev_mcx": round(prev_mcx, 2),
            }
            log.info(f"h={horizon}d | regime={regime} | ₹ change={inr_change:+.0f}")
        except Exception as e:
            log.error(f"Prediction failed for h={horizon}: {e}")
            results[horizon] = {"error": str(e)}

    return results


if __name__ == "__main__":
    cfg = load_config()
    fpath = os.path.join(cfg["paths"]["processed_data"], "features_daily.parquet")
    df = pd.read_parquet(fpath)
    predictions = predict_all_horizons(df, cfg)
    import json
    print(json.dumps(predictions, indent=2))
