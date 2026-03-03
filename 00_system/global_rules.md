# GLOBAL RULES

## Identity
- Treat this as a production ML system for commodity forecasting.
- Prioritize reproducibility, traceability, and explainability.

## Coding
- Python 3.10+. Use virtualenv / pip-tools.
- Unit tests and type checking required for production code.

## Data
- All ingestions must be logged with timestamps & source_id.
- Never overwrite raw data. Use append+versioning strategy.

## Models
- Each trained model must be tracked in model_registry.md with artifacts, seed, metrics, and data snapshot_id.

## Releases
- Every promotion to production requires passing acceptance_checklist.md.
