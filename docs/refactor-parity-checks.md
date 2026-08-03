# Refactor Parity Checks

This repository treats behavior preservation as the default during modernization work. Before changing pipeline logic, verify these workflows still produce the same outputs and schemas.

## Main Workflows
- Download and merge paper candidates: `uv run python src/data/download.py`
- LLM paper scoring: `uv run python src/data/selection/select_papers.py`
- Evidence extraction for a study: `uv run python src/run_evidence_extraction.py`
- Forest plot rendering: notebook flow in `notebooks/5.0-evidence-analysis.ipynb`

## Golden Outputs
Use `data/processed/dubhirBenchmarkingQuantizationLibraries2021/` as the baseline parity fixture.

Check these files after refactors:
- `improvement_metrics.parquet`
- `improvement_statistics_by_configuration.parquet`
- `improvement_statistics_by_precision.parquet`
- `effects_by_configuration.json`
- `effects_by_precision.json`

## Automated Gates
- Run tests: `PYTHONPATH=. .venv/bin/python -m pytest -q`
- Run lint: `UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests`

## What to Compare
- Parquet schema and column order
- Row counts and grouping keys
- Stable numeric results within small floating-point tolerance
- JSON object shape and top-level keys
- Plot layout snapshots for labels, limits, and ticks
