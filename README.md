---
title: Gold Forecast India
emoji: 🥇
colorFrom: yellow
colorTo: orange
sdk: gradio
sdk_version: "4.20.0"
app_file: app.py
pinned: false
license: mit
short_description: AI-driven gold price forecasting for Indian jewellery retailers
---

# Gold Price Forecast — Indian Jewellery Market

Production-grade ML system for Indian physical gold retailers.

**Stack**: XGBoost | 2-state HMM | GARCH(1,1) | Dynamic VaR Decision Engine

**Horizons**: 1d / 2d / 3d / 7d (direct models)

**Signal outputs**: BUY NOW | STOCK UP | HOLD | WAIT | LIQUIDATE PARTIAL

## How to Use

1. Click **Run Forecast**
2. Wait ~60-90s for first run (data fetch + model train on CPU)
3. View signal, VaR, confidence band, and per-horizon forecast table

## Architecture

```
Live Data (yfinance + FRED)
    -> Log Return Computation
    -> Feature Engineering (lags, rolling, seasonal)
    -> HMM Regime Detection (LOW/HIGH volatility)
    -> GARCH Conditional Variance Feature
    -> XGBoost Inference (regime-conditioned)
    -> Dynamic Threshold (Sharpe + VaR + regime)
    -> Signal: BUY NOW / WAIT / STOCK UP / LIQUIDATE / HOLD
```

## Disclaimer

This is a decision-support tool only. Do not base procurement decisions solely on AI signals.
Combine with your inventory levels, cash flow, and market knowledge.
