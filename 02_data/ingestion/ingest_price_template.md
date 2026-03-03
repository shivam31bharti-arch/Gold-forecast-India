# ingest_price_template.md

## Purpose
Template for price collector job.

## Steps
1. GET <api_endpoint> (use api key from secrets)
2. Validate response schema (timestamp, open/high/low/close)
3. Write to `02_data/raw/<source>/<date>/prices-<timestamp>.json` (include meta.json)
4. Normalize: convert to UTC, fill missing timestamps with NaN
5. Upsert to staging DB/table or `02_data/staging/<source>-daily.parquet`
6. Log result to ingest log
