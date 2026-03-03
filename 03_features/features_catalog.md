# FEATURES CATALOG

### Price-based
- MCX Gold futures (local benchmark)
- USD Gold (global benchmark)
- close, open, high, low (t)
- returns: r1 = (close_t - close_{t-1})/close_{t-1}
- rolling features: 7d, 14d, 30d returns/volatility

### Macro (Global & Indian)
- USD/INR exchange rate (CRITICAL)
- Import duty rate changes
- dxy, 10y yield, cpi_monthly (lagged and change)
- fed_rate_change (binary + magnitude)
- Crude oil price (affects INR)
- RBI gold reserve purchases

### Sentiment & Cycle
- Seasonal Demand Index (Wedding season Nov-Feb/Apr-May, Akshaya Tritiya, Diwali)
- daily mean FinBERT score (all gold-tagged articles)
- Global Risk Index (VIX)

### Geopolitical
- event_flag_<type> (binary) e.g. war_outbreak, sanctions_announced
- event_severity_<type> (0-1 float) aggregated from counts & sentiment
- days_since_event_<type> (integer)

### Derived
- etf_flows_GLD (net flows)
- Implied local premium (MCX vs USD gold derived)
