# MONITORING & ALERTING

Monitor:
- Data drift (feature distribution distance, e.g., KS test)
- Prediction drift (MAE vs baseline)
- Model health (latency, error rate)

Alerts:
- Trigger retrain if: rolling MAE > baseline + 2σ for 5 days
- Trigger immediate alert on missing data or ingest failure

Tools:
- Prometheus/Grafana for metrics
- Sentry for service errors
- Simple webhook to Slack/email
