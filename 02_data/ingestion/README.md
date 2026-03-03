# ingestion — Collector runbook

Each ingestion must:
1. Fetch data and store raw file under `02_data/raw/<source>/<YYYY-MM-DD>/`
2. Write a metadata record (`source`, `fetched_at`, `checksum`, `records`)
3. Run light validation (schema, timestamp monotonicity)
4. Push to staging after normalization with a version tag
