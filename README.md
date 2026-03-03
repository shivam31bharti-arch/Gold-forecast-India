# Gold Price Forecasting — Indian Jewellery Market

Production-grade ML system delivering 1/2/3/7-day gold price forecasts for Indian physical jewellery retailers.

## Architecture
- **Target**: MCX log return → ₹/10g
- **Regime**: 2-state HMM (LOW/HIGH volatility)
- **Models**: 8 XGBoost (4 horizons × 2 regimes)
- **Decision Engine**: BUY / WAIT / STOCK UP / LIQUIDATE / HOLD

## Setup

```bash
git clone <repo>
cd Gold_project
pip install -r requirements.txt
cp .env.example .env  # add FRED_API_KEY
```

## Run Pipeline

```bash
# 1. Ingest data
python data/ingest.py

# 2. Build features + regime labels
python features/engineer.py
python regime/hmm_model.py

# 3. Train models
python train/trainer.py

# 4. Validate
python validate/backtest.py

# 5. Run local inference
python inference/predictor.py

# 6. Launch dashboard
streamlit run deploy/app.py
```

## Environment Variables

```env
FRED_API_KEY=your_fred_api_key_here
```

Get FRED key free at: https://fred.stlouisfed.org/docs/api/api_key.html

## Deployment
Hosted on HuggingFace Spaces (Streamlit, CPU-only).

## Folder Structure

```
data/          ← ingestion + preprocessing
features/      ← feature engineering + seasonal
regime/        ← HMM + GARCH
train/         ← XGBoost training + Optuna tuning
inference/     ← predictor + decision engine
deploy/        ← Streamlit app
validate/      ← backtest
utils/         ← metrics + logger
models/        ← artifacts + registry
```
