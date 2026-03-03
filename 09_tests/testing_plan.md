# TESTING PLAN

1. Unit tests: run `pytest tests/unit`
2. Integration: run pipeline on `sample_snapshot/` and verify outputs
3. Performance regression: assert RMSE <= baseline_rmse * 1.05
