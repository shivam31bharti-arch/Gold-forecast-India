# TRAINING PLAN

## Target Variable
Log return: log(MCX_t / MCX_{t-1})
Output: ₹ change = MCX_{t-1} × (exp(predicted_log_return) - 1)

## Model Stack
Primary: XGBoost (per horizon: 1d, 2d, 3d, 7d) — direct multi-horizon
Regime-conditioned: Separate XGB per regime (LOW / HIGH from HMM)
Ensemble option: LightGBM + Optuna tuning as secondary

## Full Training Pipeline

### Step 0: Regime Detection
1. Fit HMM (2 states) on full daily log returns
2. Label each day: regime = LOW (0) or HIGH (1)
3. Fit GARCH(1,1) on returns → extract σ²_t as daily feature
4. Validate: COVID Mar 2020 and Russia-Ukraine Feb 2022 = high regime

### Step 1: Data Snapshot
1. Checkout data snapshot (id, date)
2. Confirm full feature matrix is available

### Step 2: Feature Build
1. Lag features: 1d, 2d, 7d price and macro lags
2. Rolling stats: 7d/14d/30d return, volatility, momentum
3. Seasonal flags: wedding_season, festival_week, akshaya_tritiya
4. Geopolitical: binary flags + severity scores + days_since decay
5. GARCH volatility σ²_t as feature
6. HMM regime flag as feature

### Step 3: Train-Test Split (Walk-Forward)
- Use walk-forward cross validation (no leakage)
- Training window: 2-year sliding window
- Test window: 3-month holdout at end

### Step 4: Train Baseline XGBoost
- One XGB per horizon (1d, 2d, 3d, 7d)
- One XGB per regime × horizon (8 models total)
- Hyperparameter tuning: Optuna (50 trials per model)

### Step 5: Evaluate
- MAPE, Directional Accuracy, Precision/Recall on up-class
- Sharpe of simulated signal per horizon
- Regime-aware split (separate metrics for LOW/HIGH regime)
- Acceptance gate: see 00_system/acceptance_checklist.md

### Step 6: Save
- Save model artifacts (.pkl) to 04_models/artifacts/
- Log to model_registry.md (id, snapshot, metrics)

### Step 7: Calibrate Decision Engine
- Run weekly_calibrate() to set threshold.json

### Step 8: Package Inference
- Dockerize inference service (FastAPI)
- CPU-only XGBoost pkl load (< 1s)
- Deploy to HuggingFace Spaces (free, permanent URL)

## Overfitting Prevention
- Walk-forward CV (no data leakage)
- L2 regularization in XGBoost (reg_lambda tuned by Optuna)
- Early stopping on validation loss
- SHAP analysis: reject features with near-zero importance

## Retraining Schedule
- Rolling 7-day retrain on 2-year sliding window
- Emergency retrain trigger: rolling MAE > baseline + 2σ for 3 consecutive days
- Regime re-label on each retrain (HMM re-fit on new data)
