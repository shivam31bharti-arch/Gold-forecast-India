# BACKTEST EXAMPLES

Example simple trading rule:
- If direction_prob > 0.65 and event_severity < 0.3 -> long for 1 day (or until signal flips)
- If direction_prob < 0.35 -> short / flatten

Compute daily PnL, cum-return, Sharpe, and max-drawdown.
Ensure transaction cost (slippage) assumptions included.
