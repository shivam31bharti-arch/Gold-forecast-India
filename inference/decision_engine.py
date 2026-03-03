"""
inference/decision_engine.py — Dynamic threshold calibration and signal generation.

Outputs one of: BUY NOW | WAIT | STOCK UP | LIQUIDATE PARTIAL | HOLD
BUY is BLOCKED if VaR (5th pct worst case) loss > ₹1000.
Threshold self-calibrates weekly from Sharpe + VaR + regime.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict

from utils.config_loader import load_config
from utils.metrics import sharpe_ratio, inr_var
from utils.logger import get_logger

log = get_logger("decision_engine")

SIGNAL_BUY      = "BUY NOW"
SIGNAL_WAIT     = "WAIT"
SIGNAL_STOCK_UP = "STOCK UP"
SIGNAL_LIQUIDATE= "LIQUIDATE PARTIAL"
SIGNAL_HOLD     = "HOLD INVENTORY"


# ── Threshold Calibration ──────────────────────────────────────────────────

def compute_dynamic_threshold(
    sharpe_14: float,
    var_14_inr: float,
    vol_regime: int,
    cfg: dict
) -> float:
    """
    Adjust decision threshold based on recent signal quality and risk.

    Args:
        sharpe_14:  Rolling 14-day Sharpe of signal returns
        var_14_inr: Rolling 14-day 5th percentile realized PnL (₹)
        vol_regime: 0 = LOW, 1 = HIGH
        cfg:        Config dict

    Returns:
        final_threshold (clipped to [0.55, 0.80])
    """
    dcfg = cfg["decision"]
    base = dcfg["base_threshold"]

    # Sharpe adjustment
    if sharpe_14 > 1.5:
        adj_sharpe = -0.05
    elif sharpe_14 < 0.5:
        adj_sharpe = +0.10
    else:
        adj_sharpe = 0.0

    # VaR penalty
    if var_14_inr < -1000:
        adj_var = +0.10
    elif var_14_inr < -500:
        adj_var = +0.05
    else:
        adj_var = 0.0

    # Regime penalty
    adj_regime = +0.05 if vol_regime == 1 else 0.0

    raw = base + adj_sharpe + adj_var + adj_regime
    final = float(np.clip(raw, dcfg["threshold_min"], dcfg["threshold_max"]))

    log.info(
        f"Threshold: base={base} + sharpe_adj={adj_sharpe} + var_adj={adj_var} "
        f"+ regime_adj={adj_regime} → final={final:.3f}"
    )
    return final


def weekly_calibrate(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Recompute and persist threshold config every Sunday.
    Reads last 14 days of realized returns to compute Sharpe + VaR.
    """
    lookback = cfg["decision"]["sharpe_lookback_days"]
    recent_lr = df["mcx_approx_logret"].dropna().tail(lookback).values

    if len(recent_lr) < 5:
        log.warning("Not enough recent data for calibration. Using default threshold.")
        threshold = cfg["decision"]["base_threshold"]
        sharpe_14 = 0.0
        var_14_inr = 0.0
    else:
        sharpe_14 = sharpe_ratio(recent_lr)
        prev_mcx = df["mcx_approx"].dropna().iloc[-1]
        var_14_inr = inr_var(recent_lr, prev_mcx, percentile=5.0)
        vol_regime = int(df["regime"].dropna().iloc[-1]) if "regime" in df.columns else 0
        threshold = compute_dynamic_threshold(sharpe_14, var_14_inr, vol_regime, cfg)

    config_payload = {
        "final_threshold": threshold,
        "sharpe_14": round(sharpe_14, 4),
        "var_14_inr": round(var_14_inr, 2),
        "updated_at": datetime.now().isoformat(),
    }

    path = cfg["paths"]["threshold_config"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config_payload, f, indent=2)
    log.info(f"Threshold config saved: {config_payload}")
    return config_payload


def load_threshold(cfg: dict) -> float:
    """Load persisted threshold. Fall back to base if not found."""
    path = cfg["paths"]["threshold_config"]
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return float(data.get("final_threshold", cfg["decision"]["base_threshold"]))
    return float(cfg["decision"]["base_threshold"])


# ── Signal Logic ───────────────────────────────────────────────────────────

def generate_signal(predictions: Dict[int, dict], df: pd.DataFrame, cfg: dict) -> dict:
    """
    Apply decision engine to model predictions.

    Args:
        predictions: {horizon: {...predicted_log_return, predicted_inr_change, direction_prob, regime}}
        df:          Feature DataFrame (for VaR and Sharpe computation)
        cfg:         Config dict

    Returns:
        Full signal dict with recommendation, confidence band, VaR, etc.
    """
    dcfg = cfg["decision"]
    threshold = load_threshold(cfg)
    block_var_threshold = dcfg["var_block_threshold_inr"]

    # Compute VaR from recent returns
    recent_lr = df["mcx_approx_logret"].dropna().tail(30).values
    prev_mcx = df["mcx_approx"].dropna().iloc[-1]
    var_inr = inr_var(recent_lr, prev_mcx, percentile=5.0) if len(recent_lr) >= 10 else -9999.0

    # Confidence band (±1 std of recent 30d returns, in ₹)
    std_lr = float(np.std(recent_lr)) if len(recent_lr) >= 2 else 0.01
    band_inr = prev_mcx * std_lr

    # Use 1d and 7d predictions
    pred_1d = predictions.get(1, {})
    pred_7d = predictions.get(7, {})

    if "error" in pred_1d or "error" in pred_7d:
        return {
            "signal": SIGNAL_HOLD,
            "reason": "Prediction unavailable — model error.",
            "threshold": threshold,
            "var_inr": round(var_inr, 2),
            "band_inr": round(band_inr, 2),
        }

    dir_prob_1d = pred_1d.get("direction_prob", 0.5)
    dir_prob_7d = pred_7d.get("direction_prob", 0.5)
    move_7d_pct = pred_7d.get("predicted_log_return", 0) * 100
    move_1d_inr = pred_1d.get("predicted_inr_change", 0)
    regime = pred_1d.get("regime", 0)

    # ── STOCK UP ──
    if (move_7d_pct > dcfg["stock_up_min_return"] * 100
            and dir_prob_1d > threshold
            and var_inr >= block_var_threshold):
        signal = SIGNAL_STOCK_UP
        reason = (f"7d forecast: +{move_7d_pct:.1f}% expected rise. "
                  f"1d direction confidence {dir_prob_1d:.0%}. VaR acceptable.")

    # ── BUY NOW ──
    elif (dir_prob_1d > threshold
          and move_1d_inr > 0
          and var_inr >= block_var_threshold):
        signal = SIGNAL_BUY
        reason = (f"1d confidence {dir_prob_1d:.0%} > threshold {threshold:.0%}. "
                  f"Expected +₹{move_1d_inr:,.0f}. VaR = ₹{var_inr:,.0f} (acceptable).")

    # ── LIQUIDATE ──
    elif dir_prob_1d < (1 - threshold) and var_inr < block_var_threshold:
        signal = SIGNAL_LIQUIDATE
        reason = (f"High downside confidence {1-dir_prob_1d:.0%}. "
                  f"VaR = ₹{var_inr:,.0f} — significant loss risk.")

    # ── WAIT ──
    elif var_inr < block_var_threshold:
        signal = SIGNAL_WAIT
        reason = (f"VaR = ₹{var_inr:,.0f} exceeds ₹{block_var_threshold:,.0f} limit. "
                  f"Hold off until volatility settles.")

    # ── HOLD ──
    else:
        signal = SIGNAL_HOLD
        reason = (f"No strong directional signal. "
                  f"Dir confidence {dir_prob_1d:.0%} within neutral band.")

    return {
        "signal":             signal,
        "reason":             reason,
        "regime":             regime,
        "regime_label":       "HIGH VOLATILITY" if regime else "LOW VOLATILITY",
        "threshold_used":     round(threshold, 3),
        "var_inr":            round(var_inr, 0),
        "confidence_band_inr":round(band_inr, 0),
        "pred_1d_inr":        round(move_1d_inr, 0),
        "pred_7d_pct":        round(move_7d_pct, 2),
        "dir_prob_1d":        round(dir_prob_1d, 3),
        "dir_prob_7d":        round(dir_prob_7d, 3),
        "current_mcx_approx": round(prev_mcx, 0),
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M IST"),
    }


if __name__ == "__main__":
    from inference.predictor import predict_all_horizons
    cfg = load_config()
    fpath = os.path.join(cfg["paths"]["processed_data"], "features_daily.parquet")
    df = pd.read_parquet(fpath)

    # Calibrate
    weekly_calibrate(df, cfg)

    # Predict
    preds = predict_all_horizons(df, cfg)

    # Signal
    result = generate_signal(preds, df, cfg)
    print(json.dumps(result, indent=2))
