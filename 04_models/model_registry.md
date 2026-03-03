# MODEL REGISTRY

Record format:
- model_id: <uuid>
- tag: tft_v1
- created_at: ISO timestamp
- data_snapshot_id:
- hyperparams: yaml
- metrics: json
- artifact_uri: s3://... or ./artifacts/<model_id>.pkl
- notes

Add a new entry for every trained model.
