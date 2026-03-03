# REGIME DETECTION

## Decision
Use HMM (2-state) for regime labeling + GARCH(1,1) as a volatility feature.
This combination is validated in peer-reviewed research on MCX gold forecasting.

## Why HMM Wins Over Other Methods

| Method | Verdict |
|---|---|
| HMM (2-state on returns) | ✅ BEST — captures structural breaks (COVID, Russia-Ukraine), probabilistic, persistent |
| GARCH(1,1) | ✅ Use as FEATURE, not classifier — models volatility clustering perfectly |
| Rolling 30d StdDev | ⚠️ Baseline fallback only — brittle, no persistence |
| K-Means clustering | ❌ Rejected — ignores temporal dependency, regimes flip randomly |

## Verified by Research
- HMM on daily returns effectively captures COVID + geopolitical regime breaks
  (Baum-Welch estimation, Viterbi state inference)
- GARCH-type models are specifically validated on MCX gold (EGARCH, TGARCH)
  capturing the leverage effect (negative shocks > positive shocks)
- Studies confirm gold exhibits strong volatility clustering — GARCH fits naturally

## Implementation Plan
```
Step 1: Fit HMM (2 states) on daily log returns of MCX gold
        Library: hmmlearn (Python)
        Input: [log_return_t, realized_vol_t]  (bivariate emission)

Step 2: Validate alignment:
        State 1 should cover → COVID Mar 2020 spike, Russia-Ukraine Feb 2022

Step 3: Label each day in history with: regime = LOW (0) or HIGH (1)

Step 4: Train separate models:
        XGB_model_low  → trained on regime=0 days
        XGB_model_high → trained on regime=1 days

Step 5: At inference time:
        Detect current regime (run HMM on last 30 days)
        Select corresponding model
```

## Fallback If HMM Unstable (<1200 rows, parameter instability)
```
regime = 1 if rolling_std_30d > 75th_percentile_of_all_rolling_stds else 0
```

## GARCH as Feature (NOT classifier)
```
Fit GARCH(1,1) on full return series
Extract conditional variance σ²_t daily
Add as feature: garch_vol_t
Also derive: high_garch_flag = 1 if σ²_t > 90th percentile
```

## Recommended Library Stack
- `hmmlearn` — HMM fitting
- `arch` — GARCH fitting (Python, widely used for MCX-style data)
