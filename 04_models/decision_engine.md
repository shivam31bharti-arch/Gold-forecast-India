# DECISION ENGINE

## Output Signals
The model does NOT output raw ₹ prices. It outputs ONE of:

| Signal | Condition |
|---|---|
| BUY NOW | 2d + 7d forecast both rise > threshold, VaR acceptable |
| WAIT X DAYS | Directional confidence < threshold OR VaR > ₹1000 loss |
| STOCK UP | 7d gradual rise predicted, 1d flat (accumulate before spike) |
| LIQUIDATE PARTIAL | High confidence of 7d drop, VaR > ₹1500 loss |
| HOLD INVENTORY | Default — no clear signal |

## Dynamic Threshold Calibration (Self-Calibrating Weekly)

### Core Variables
```python
sharpe_14    # rolling 14-day Sharpe of past signal returns
var_14       # rolling 5th percentile of realized PnL (₹ terms)
vol_regime   # HMM state: LOW (0) or HIGH (1)
```

### Base Threshold
```python
base_threshold = 0.60
```

### Sharpe Adjustment
```python
if sharpe_14 > 1.5:
    threshold_adj_sharpe = -0.05   # strong signal → lower bar to BUY
elif sharpe_14 < 0.5:
    threshold_adj_sharpe = +0.10   # weak signal → be more conservative
else:
    threshold_adj_sharpe = 0.0
```

### VaR Penalty (critical for ₹1000 margin protection)
```python
if var_14 < -1000:                 # recent worst-case loss > ₹1000
    threshold_adj_var = +0.10
elif var_14 < -500:
    threshold_adj_var = +0.05
else:
    threshold_adj_var = 0.0
```

### Volatility Regime Penalty
```python
if vol_regime == HIGH:
    threshold_adj_vol = +0.05      # high regime → require stronger signal
else:
    threshold_adj_vol = 0.0
```

### Final Threshold
```python
final_threshold = base_threshold + threshold_adj_sharpe + threshold_adj_var + threshold_adj_vol
final_threshold = clip(final_threshold, min=0.55, max=0.80)
```

### Decision Logic (1-day model)
```python
if direction_prob > final_threshold:
    if expected_move > 0 and var_14 > -1000:
        signal = "BUY NOW"
    else:
        signal = "WAIT"

elif direction_prob < (1 - final_threshold):
    signal = "LIQUIDATE PARTIAL"

else:
    signal = "HOLD"
```

### STOCK UP Logic (7-day horizon)
```python
if (7d_expected_return > +0.015          # predicted +1.5% rise
    and 1d_direction_prob > final_threshold
    and var_14 > -1000):
        signal = "STOCK UP"
```

## Weekly Self-Calibration Loop (No Human Needed)
```python
# Run every Sunday 11:59 PM IST
def weekly_calibrate():
    sharpe_14 = compute_rolling_sharpe(last_14_days_signals)
    var_14    = compute_5th_percentile_pnl(last_14_days_pnl)
    vol_regime = hmm_model.predict(last_30_days_returns)[-1]

    config = {
        "final_threshold": compute_threshold(sharpe_14, var_14, vol_regime),
        "updated_at": datetime.now().isoformat()
    }
    save_json(config, "config/threshold.json")
```

## Output Format per Signal (Displayed to Seller)
```
Signal:          BUY NOW
Horizon:         7-day
Expected Move:   +₹1,200 per 10g (+1.4%)
Confidence:      72%
Worst Case (VaR): -₹800 per 10g
Regime:          LOW volatility
Recommendation:  Stock up for wedding season inventory.
```

## Design Principles
- Sharpe ensures signal QUALITY before BUY is triggered
- VaR prevents catastrophic wrong calls (never BUY into >₹1000 downside risk)
- Regime flag suppresses aggressive calls during chaos (COVID / war spikes)
- Clipping (0.55–0.80) avoids extreme over-adjustment in edge cases
