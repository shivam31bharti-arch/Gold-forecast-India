# MODEL ARCHITECTURE

## Overview
We need multi-output predictions (per horizon — direct forecasting approach):
- 1-day log return (Model A)
- 2-day log return (Model B)
- 3-day log return (Model C)
- 7-day log return (Model D)
- All converted to ₹ change for decision engine output

## Target Variable
Train on log returns: log(MCX_t / MCX_{t-1})
Output in ₹: MCX_{t-1} × (exp(predicted_log_return) - 1)

## Primary Target: MCX Gold (₹/10g)
Do NOT model USD Gold + USD/INR separately and combine.
MCX already encodes USD price × exchange rate × import duty + local premium.
USD Gold, USD/INR, and import duty are FEATURES, not components.

## Recommended Model: XGBoost (per-horizon, direct)
- Tabular, daily resolution → XGBoost outperforms LSTM/TFT on parsimony
- Explainable via SHAP
- Fast training (no GPU needed), CPU-only inference
- Use LightGBM + Optuna tuning as secondary

## Regime Handling
- Detect LOW/HIGH volatility regimes (HMM or GARCH threshold)
- Train separate XGBoost models per regime
- Or pass regime_flag as feature to single model
- Never mix COVID-era data with calm periods in same training window

## Risk Outputs (Per Prediction)
- Median prediction (expected move)
- 75th percentile (upside)
- 5th percentile VaR (worst-case loss) — primary risk signal

## Optional: TFT (Ensemble Only)
- Only justified if: 5yr+ clean daily dataset + GPU for training available
- Train offline (Google Colab), commit model artifact to repo
- Use as stacked ensemble member, not primary

## Multi-head Loss (if TFT used)
- Shared encoder → regression head (MSE + quantile loss) + classification head (cross-entropy)
- Loss = weighted sum; tune weights to balance regression vs directional accuracy

## Overfitting Prevention
- Time-series cross validation (rolling window split, walk-forward)
- Regularization: L2, dropout
- Early stopping
- Gradient clipping + layer normalization (for TFT/LSTM only)
- SHAP analysis for feature dominance review
