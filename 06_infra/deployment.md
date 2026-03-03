# DEPLOYMENT

Options:
- Serve model as REST API in Docker (FastAPI + uvicorn)
- Batch scoring via Airflow or scheduled Lambda/K8s job
- Model registry and artifacts in S3 or modelstore

Inference requirements:
- Single-request latency < 500ms (if serving real-time)
- Scoring endpoint accepts data_snapshot_id and returns price, direction, CI

Containerization:
- Dockerfile should pin python and reproduce training environment
