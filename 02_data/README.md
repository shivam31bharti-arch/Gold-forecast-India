# 02_data — Data & ingestion

Structure:
- ingestion/: templates & runbooks for collectors
- raw/: immutable raw dumps per source, date-partitioned
- staging/: lightly cleaned, normalized data
- processed/: feature-aligned time-series ready for modeling

Principles:
- immutable raw layer
- metadata + provenance for every file
- dataset snapshot IDs for each model training
