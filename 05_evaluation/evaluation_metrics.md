# EVALUATION METRICS

Statistical:
- MAPE (7-day) < 1.5% is target. Absolute precise ₹ prediction is unrealistic.
- Quantile coverage (95% CI)

Directional:
- Directional accuracy (hit rate) ≥ 60%

Economic:
- Sharpe ratio of trading signal > 1.2
- Margin protection (simulated avoidance of negative margin days)

Decision Engine:
- Instead of raw price, evaluates success of final outputs: "BUY NOW", "WAIT 2 DAYS", "STOCK UP".

Regime-aware:
- Compute all metrics per regime (low-volatility / high-volatility)
- Track stability over rolling retrain windows (time-series cross validation, walk-forward validation)

Operational:
- Inference latency
- Bias/Variance protection hooks enabled (early stopping, L2/dropout, gradient clipping for TFT)
