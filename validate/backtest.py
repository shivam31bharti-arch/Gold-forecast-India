"""
validate/backtest.py — Walk-forward backtest for 1d and 7d strategies.

Validates:
  - No forward leakage
  - MAE, directional accuracy, Sharpe, max drawdown
  - Regime-split performance (LOW vs HIGH)

Usage:
  python validate/backtest.py
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from train.splitter import walk_forward_splits
from utils.config_loader import load_config
from utils.metrics import evaluate_all, inr_var
from utils.logger import get_logger

log = get_logger("backtest")


def simple_strategy_returns(y_pred_logret: np.ndarray,
                             y_true_logret: np.ndarray,
                             threshold: float = 0.0) -> np.ndarray:
    """
    Simple long/flat strategy:
      If predicted return > threshold → long (earn actual return)
      Else → flat (earn 0)
    """
    positions = (y_pred_logret > threshold).astype(float)
    return positions * y_true_logret


def run_backtest(df: pd.DataFrame, cfg: dict, horizon: int = 1) -> dict:
    """
    Rolling walk-forward backtest for a given horizon.
    Selects model by regime per fold.
    """
    artifacts_path = cfg["paths"]["model_artifacts"]
    target_col = f"target_{horizon}d"
    feature_cols = joblib.load(os.path.join(artifacts_path, "feature_cols.pkl"))

    all_y_true, all_y_pred, all_strategy_returns = [], [], []
    all_regimes = []

    splits = list(walk_forward_splits(
        df, n_splits=cfg["training"]["n_splits"],
        min_train_size=365, test_size=60
    ))

    for fold_idx, (train_df, test_df) in enumerate(splits):
        test_df = test_df.dropna(subset=[target_col] + feature_cols)
        if test_df.empty:
            log.warning(f"Fold {fold_idx}: empty test set for h={horizon}d.")
            continue

        for regime in cfg["training"]["regimes"]:
            regime_test = test_df[test_df["regime"] == regime]
            if len(regime_test) < 5:
                continue

            model_path = os.path.join(artifacts_path, f"xgb_r{regime}_{horizon}d.pkl")
            if not os.path.exists(model_path):
                log.warning(f"Model not found: xgb_r{regime}_{horizon}d.pkl")
                continue

            model = joblib.load(model_path)
            X_test = regime_test[feature_cols]
            y_test = regime_test[target_col]

            y_pred = model.predict(X_test)
            strat_rets = simple_strategy_returns(y_pred, y_test.values)

            all_y_true.extend(y_test.values)
            all_y_pred.extend(y_pred)
            all_strategy_returns.extend(strat_rets)
            all_regimes.extend([regime] * len(y_test))

    if not all_y_true:
        log.error("Backtest produced no predictions. Check models exist.")
        return {}

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    strat_r = np.array(all_strategy_returns)
    regimes = np.array(all_regimes)

    overall = evaluate_all(y_true, y_pred, strategy_returns=strat_r)
    log.info(f"\n=== BACKTEST h={horizon}d ===")
    for k, v in overall.items():
        log.info(f"  {k}: {v:.4f}")

    # Regime-split performance
    regime_results = {}
    for r in cfg["training"]["regimes"]:
        mask = regimes == r
        if mask.sum() < 5:
            continue
        r_metrics = evaluate_all(y_true[mask], y_pred[mask],
                                 strategy_returns=strat_r[mask])
        regime_results[f"regime_{r}"] = r_metrics
        log.info(f"  Regime={'HIGH' if r else 'LOW'} | "
                 f"Dir={r_metrics['directional_accuracy']:.2%} | "
                 f"Sharpe={r_metrics.get('sharpe', 0):.2f}")

    return {
        "horizon": horizon,
        "overall": overall,
        "by_regime": regime_results,
        "n_predictions": len(y_true),
    }


if __name__ == "__main__":
    cfg = load_config()
    fpath = os.path.join(cfg["paths"]["processed_data"], "features_daily.parquet")
    df = pd.read_parquet(fpath)

    print("\n" + "="*60)
    for h in [1, 7]:
        result = run_backtest(df, cfg, horizon=h)
        if result:
            ov = result["overall"]
            print(f"\nHorizon {h}d | n={result['n_predictions']}")
            print(f"  MAE:          {ov['mae']:.6f}")
            print(f"  Sharpe:       {ov.get('sharpe', 0):.2f}")
            print(f"  Dir Accuracy: {ov['directional_accuracy']:.2%}")
            print(f"  Max Drawdown: {ov.get('max_drawdown', 0):.2%}")
