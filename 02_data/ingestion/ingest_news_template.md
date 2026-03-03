# ingest_news_template.md

## Purpose
Template for news+sentiment ingestion.

## Steps
1. Query news APIs with keywords: "gold", "precious metals", "fed", "inflation"
2. Persist raw article JSON to `02_data/raw/news/<source>/<date>/`
3. Extract fields: title, body, published_at, source, url
4. Run NLP: extract sentiment (FinBERT), named entities, topic tags
5. Save processed records to `02_data/processed/news.parquet` with embedding_id if used
