# ACCEPTANCE CHECKLIST

Before promoting model → staging/production:

- ✅ Data lineage documented and snapshot saved
- ✅ Repro training script available (seeded)
- ✅ Unit tests ≥ 80% and integration tests passing
- ✅ Evaluation metrics: RMSE, directional accuracy, Sharpe (baseline thresholds met)
- ✅ Model performance across regimes tested (stress windows)
- ✅ Explainability report (SHAP or TFT variable importance)
- ✅ Monitoring hooks (drift detection, latency, alerts) configured
