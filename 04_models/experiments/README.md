# experiments — How to log experiments

Each experiment folder should contain:
- config.yaml (data snapshot id, hyperparams)
- train.log
- metrics.json (RMSE, MAE, directional_acc, sharpe, max_dd)
- model artifact reference
Use MLflow or a simple JSON-based registry.
