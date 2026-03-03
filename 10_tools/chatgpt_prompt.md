**You are a senior quantitative system architect specializing in commodity forecasting and financial decision systems.**

We are building a gold price prediction engine specifically for an Indian jewellery seller.

### Business Context:
- Seller purchases physical gold every 1–3 days from distributor.
- ₹1000–₹2000 movement per 10g significantly impacts profit margin.
- Needs 1-day, 2-day, 3-day, and 7-day forecasts.
- Needs buy/wait recommendations, not just raw price predictions.
- Gold pricing in India depends on:
    - USD gold price
    - USD/INR exchange rate
    - Import duty
    - MCX gold futures
    - RBI policy changes
    - Seasonal demand (wedding season, festivals)
    - Global risk (VIX, war, sanctions)

### Data Sources (free):
- FreeGoldAPI (historical)
- FMP / AlphaVantage (spot price)
- FRED (DXY, 10Y yield, CPI)
- USD/INR exchange rate
- MCX gold futures
- Marketaux / NewsAPI / GDELT (sentiment + geopolitical)
- ETF flows (GLD)
- RBI gold purchase announcements
- Crude oil prices

### System Requirements:

1. **Explain how you will design:**
   - Data cleaning pipeline
   - Missing data handling
   - Feature engineering (lag features, rolling stats, volatility clusters, seasonal encoding)
   - Scaling (standardization vs robust scaling vs no scaling for trees)
   - Sentiment aggregation logic
   - Geopolitical event encoding (binary + severity + decay modeling)

2. **What evaluation metrics will you use and why:**
   - RMSE / MAE / MAPE
   - Directional accuracy
   - Precision/Recall for "price rise" class
   - Sharpe ratio of signal
   - Regime-aware performance analysis

3. **What expected realistic accuracy targets should we aim for in:**
   - 1-day horizon
   - 7-day horizon

4. **How will you prevent:**
   - Overfitting
   - Underfitting
   - Bias-variance imbalance
   - Exploding gradients (if using LSTM/TFT)
   - Data leakage in time series cross-validation

5. **What additional unbiased data sources should we include to improve robustness for Indian gold price modeling?**

6. **Deployment:**
   - We need this deployed via GitHub repository with a working public URL.
   - Suggest best deployment option (HuggingFace Spaces, Streamlit Cloud, Render, etc.)
   - Confirm whether GPU is required (training vs inference separation). CPU inference is preferred.

7. **Decision Engine:**
   Design logic to output:
   - BUY NOW
   - WAIT X DAYS
   - STOCK UP (expected rise)
   - LIQUIDATE PARTIAL
   - HOLD INVENTORY
   Include confidence band and expected price change magnitude.

8. **Precision Requirement:**
   Even ₹1000 movement matters.
   Explain realistically how precise such a model can be and what uncertainty bands we should communicate.

*Respond technically and critically. Focus on real-world viability, not theoretical perfection.*
