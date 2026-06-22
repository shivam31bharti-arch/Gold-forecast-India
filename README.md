---
title: Gold Forecast India
emoji: "🥇"
colorFrom: yellow
colorTo: orange
sdk: gradio
sdk_version: "4.20.0"
app_file: app.py
pinned: false
license: mit
short_description: AI-driven gold price forecasting for Indian jewellery retailers
---

# Gold Forecast India

Gold Forecast India is a forecasting and decision-support system for Indian jewellery retailers. It combines macro and market data ingestion, feature engineering, volatility regime detection, XGBoost forecasting, and a rule-based decision engine to produce inventory actions such as `BUY NOW`, `WAIT`, and `STOCK UP`.

## Stack

- XGBoost
- 2-state HMM regime detection
- GARCH volatility features
- Gradio UI for Hugging Face Spaces
- Yahoo Finance and FRED sourced market inputs

## Forecast Horizons

- 1 day
- 2 day
- 3 day
- 7 day

## Decision Outputs

- `BUY NOW`
- `STOCK UP`
- `HOLD INVENTORY`
- `WAIT`
- `LIQUIDATE PARTIAL`

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Optional:

- Set `FRED_API_KEY` if you want full FRED-backed macro ingestion.

## Pipeline

```text
Market Data
-> preprocessing
-> engineered features
-> HMM regime detection
-> GARCH volatility features
-> XGBoost horizon models
-> threshold calibration
-> inventory signal
```

## Project Entry Points

- `app.py` - Gradio app for Hugging Face Spaces and local UI
- `run_pipeline.py` - end-to-end pipeline runner
- `deploy/app.py` - alternate deployment app

## Maintenance Notes

- The deployment config now correctly identifies the UI framework as Gradio.
- First runs are slower because the app may ingest data and train CPU models before caching artifacts.
- Generated artifacts and parquet outputs are expected to live under the configured `data/` and `models/` paths.

## Disclaimer

This repository provides decision support, not financial advice. Use it alongside live procurement constraints, inventory position, and human judgment.
