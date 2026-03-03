"""
run_pipeline.py — Master script to execute the full pipeline end-to-end.

Usage:
  python run_pipeline.py [--phase ingest|features|train|backtest|serve]
  python run_pipeline.py           # run all phases
"""
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_loader import load_config
from utils.logger import get_logger
log = get_logger("pipeline")


def run_ingest(cfg):
    from data.ingest import ingest_all, save_raw
    log.info("=== PHASE 1: DATA INGESTION ===")
    df = ingest_all(cfg)
    save_raw(df, cfg)
    return df


def run_preprocess(cfg):
    import pandas as pd
    from data.preprocess import preprocess, save_processed
    log.info("=== PHASE 2: PREPROCESSING ===")
    raw = pd.read_parquet(os.path.join(cfg["paths"]["raw_data"], "raw_daily.parquet"))
    df = preprocess(raw, cfg)
    save_processed(df, cfg)
    return df


def run_features(cfg):
    import pandas as pd
    from features.engineer import build_features, save_features
    from regime.hmm_model import run_hmm_pipeline
    from regime.garch_model import add_garch_features
    log.info("=== PHASE 3: FEATURE ENGINEERING + REGIME DETECTION ===")
    proc = pd.read_parquet(os.path.join(cfg["paths"]["processed_data"], "processed_daily.parquet"))
    df = build_features(proc, cfg)
    df = run_hmm_pipeline(df, cfg)
    df = add_garch_features(df, cfg)
    save_features(df, cfg)
    return df


def run_train(cfg):
    import pandas as pd
    from train.trainer import train_all_models
    log.info("=== PHASE 4: MODEL TRAINING (8 XGBoost models) ===")
    df = pd.read_parquet(os.path.join(cfg["paths"]["processed_data"], "features_daily.parquet"))
    registry = train_all_models(df, cfg)
    log.info(f"Training complete — {len(registry)} models trained.")
    return registry


def run_backtest(cfg):
    import pandas as pd
    from validate.backtest import run_backtest
    log.info("=== PHASE 5: BACKTEST ===")
    df = pd.read_parquet(os.path.join(cfg["paths"]["processed_data"], "features_daily.parquet"))
    for h in [1, 7]:
        run_backtest(df, cfg, horizon=h)


def run_inference(cfg):
    import pandas as pd, json
    from inference.predictor import predict_all_horizons
    from inference.decision_engine import generate_signal, weekly_calibrate
    log.info("=== PHASE 6: INFERENCE + SIGNAL ===")
    df = pd.read_parquet(os.path.join(cfg["paths"]["processed_data"], "features_daily.parquet"))
    weekly_calibrate(df, cfg)
    preds = predict_all_horizons(df, cfg)
    signal = generate_signal(preds, df, cfg)
    print(json.dumps(signal, indent=2))
    return signal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold Forecasting Pipeline")
    parser.add_argument("--phase", choices=["ingest", "features", "train", "backtest", "serve"],
                        default=None, help="Run specific phase only")
    args = parser.parse_args()

    cfg = load_config()

    if args.phase == "ingest":
        run_ingest(cfg)
        run_preprocess(cfg)
    elif args.phase == "features":
        run_features(cfg)
    elif args.phase == "train":
        run_train(cfg)
    elif args.phase == "backtest":
        run_backtest(cfg)
    elif args.phase == "serve":
        run_inference(cfg)
    else:
        # Full pipeline
        log.info("Running full pipeline end-to-end...")
        run_ingest(cfg)
        run_preprocess(cfg)
        run_features(cfg)
        run_train(cfg)
        run_backtest(cfg)
        run_inference(cfg)
        log.info("=== PIPELINE COMPLETE ===")
