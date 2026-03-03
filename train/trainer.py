"""
train/trainer.py — Train 8 XGBoost models: 4 horizons × 2 regimes.

Architecture:
  For each regime in [LOW=0, HIGH=1]:
    For each horizon in [1, 2, 3, 7]:
      XGBoost regression on log return target
      XGBoost classification on direction (up/down)

Models saved to:  models/artifacts/xgb_{regime}_{horizon}d.pkl
SHAP values saved to: models/artifacts/shap_{regime}_{horizon}d.npy
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import json
import shap
from sklearn.metrics import accuracy_score
from train.splitter import final_train_test_split
from utils.config_loader import load_config
from utils.metrics import evaluate_all
from utils.logger import get_logger

log = get_logger("trainer")


def get_feature_cols(df: pd.DataFrame, horizons: list) -> list:
    """Return feature columns — exclude raw prices, targets, and metadata."""
    exclude_prefixes = ("gold_usd", "usdinr", "dxy", "crude_usd", "mcx_approx",
                        "CPIAUCSL", "DGS10", "outlier_flag")
    excluded_exact = {"month", "day_of_week", "day_of_year", "quarter"}
    target_cols = set()
    for h in horizons:
        target_cols.add(f"target_{h}d")
        target_cols.add(f"target_{h}d_dir")

    cols = []
    for c in df.columns:
        if c in excluded_exact or c in target_cols:
            continue
        if any(c.startswith(p) and not any(s in c for s in ["logret", "lag", "rmean", "rstd"])
               for p in ("gold_usd", "usdinr", "dxy", "crude_usd", "mcx_approx")):
            continue
        cols.append(c)
    return cols


def train_single_model(X_train: pd.DataFrame, y_train: pd.Series,
                       params: dict, early_stopping_rounds: int = 30) -> xgb.XGBRegressor:
    """Train one XGBoost regressor. Return fitted model."""
    model = xgb.XGBRegressor(**params, early_stopping_rounds=early_stopping_rounds)
    # Use 10% of training as internal early-stop val (no leakage — all in past)
    val_size = max(1, int(len(X_train) * 0.10))
    X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
    y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model


def compute_shap(model: xgb.XGBRegressor, X: pd.DataFrame, save_path: str):
    """Compute and save SHAP values."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        np.save(save_path, shap_values)
        log.info(f"SHAP values saved: {save_path}")
    except Exception as e:
        log.warning(f"SHAP computation failed: {e}")


def train_all_models(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Train 8 XGBoost models (4 horizons × 2 regimes).
    Returns dict mapping (regime, horizon) → fitted model.
    """
    tcfg = cfg["training"]
    horizons = tcfg["horizons"]
    regimes = tcfg["regimes"]
    base_params = dict(tcfg["xgb_base_params"])
    artifacts_path = cfg["paths"]["model_artifacts"]
    os.makedirs(artifacts_path, exist_ok=True)

    # Feature columns (same for all models)
    feature_cols = get_feature_cols(df, horizons)
    log.info(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")

    registry = {}

    for regime in regimes:
        regime_df = df[df["regime"] == regime].copy()
        if len(regime_df) < 100:
            log.warning(f"Regime {regime} has only {len(regime_df)} rows. Skipping.")
            continue

        train_df, test_df = final_train_test_split(regime_df, test_fraction=0.15)
        log.info(f"Regime {regime} | Train: {len(train_df)} | Test: {len(test_df)}")

        for horizon in horizons:
            target_col = f"target_{horizon}d"
            if target_col not in df.columns:
                log.warning(f"Missing target column: {target_col}")
                continue

            # Filter to rows where target is available
            tr = train_df.dropna(subset=[target_col] + feature_cols)
            te = test_df.dropna(subset=[target_col] + feature_cols)

            if len(tr) < 50:
                log.warning(f"Too few training rows for regime={regime}, h={horizon}d. Skipping.")
                continue

            X_train, y_train = tr[feature_cols], tr[target_col]
            X_test,  y_test  = te[feature_cols], te[target_col]

            log.info(f"Training: regime={regime}, horizon={horizon}d | "
                     f"train={len(X_train)}, test={len(X_test)}")

            model = train_single_model(X_train, y_train, base_params,
                                       tcfg["early_stopping_rounds"])

            # Evaluate
            y_pred = model.predict(X_test)
            metrics = evaluate_all(y_test.values, y_pred)
            direction_pred = (y_pred > 0).astype(int)
            direction_true = (y_test.values > 0).astype(int)
            metrics["dir_accuracy"] = float(accuracy_score(direction_true, direction_pred))

            log.info(f"regime={regime} h={horizon}d | MAE={metrics['mae']:.6f} | "
                     f"Dir={metrics['dir_accuracy']:.2%}")

            # Save
            model_key = f"xgb_r{regime}_{horizon}d"
            model_path = os.path.join(artifacts_path, f"{model_key}.pkl")
            joblib.dump(model, model_path)
            log.info(f"Model saved: {model_path}")

            # SHAP on test set
            shap_path = os.path.join(artifacts_path, f"shap_r{regime}_{horizon}d.npy")
            compute_shap(model, X_test, shap_path)

            registry[(regime, horizon)] = {
                "model_key": model_key,
                "model_path": model_path,
                "feature_cols": feature_cols,
                "metrics": metrics,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            }

    # Save feature list for inference
    joblib.dump(feature_cols, os.path.join(artifacts_path, "feature_cols.pkl"))

    # Save registry JSON
    reg_path = cfg["paths"]["model_registry"]
    os.makedirs(os.path.dirname(reg_path), exist_ok=True)
    serializable = {f"regime{k[0]}_h{k[1]}d": v for k, v in registry.items()}
    with open(reg_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    log.info(f"Registry saved: {reg_path}")

    return registry


if __name__ == "__main__":
    cfg = load_config()
    fpath = os.path.join(cfg["paths"]["processed_data"], "features_daily.parquet")
    df = pd.read_parquet(fpath)

    registry = train_all_models(df, cfg)

    print("\n=== TRAINING COMPLETE ===")
    for key, info in registry.items():
        m = info["metrics"]
        print(f"  regime={key[0]} h={key[1]}d | "
              f"MAE={m['mae']:.6f} | Dir={m.get('dir_accuracy',0):.2%}")
