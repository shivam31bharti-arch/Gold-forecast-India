"""
utils/metrics.py — Evaluation metrics for the gold forecasting system.
"""
import numpy as np
import pandas as pd
from typing import Dict


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of times predicted direction matches actual direction."""
    correct = np.sign(y_true) == np.sign(y_pred)
    return float(np.mean(correct))


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.03) -> float:
    """Annualised Sharpe ratio of a daily return series."""
    if len(returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(252))


def max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Max drawdown from peak to trough."""
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / np.where(peak == 0, 1, peak)
    return float(np.min(drawdown))


def value_at_risk(returns: np.ndarray, percentile: float = 5.0) -> float:
    """5th percentile VaR of a return distribution."""
    return float(np.percentile(returns, percentile))


def inr_var(log_returns: np.ndarray, prev_mcx_price: float, percentile: float = 5.0) -> float:
    """VaR in ₹ per 10g terms from log returns."""
    worst_log_return = np.percentile(log_returns, percentile)
    pnl = prev_mcx_price * (np.exp(worst_log_return) - 1)
    return float(pnl)


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray,
                 strategy_returns: np.ndarray = None) -> Dict[str, float]:
    """Compute all metrics in one call. Returns a dict."""
    result = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }
    if strategy_returns is not None:
        cum_ret = np.cumprod(1 + strategy_returns)
        result["sharpe"] = sharpe_ratio(strategy_returns)
        result["max_drawdown"] = max_drawdown(cum_ret)
        result["var_5pct"] = value_at_risk(strategy_returns)
    return result
