# Repository Guidelines

## Project Structure & Module Organization
Core Python code lives in `src/`. Use `src/data/` for data acquisition, paper selection, and extraction utilities, `src/forestplot/` for plotting helpers, and top-level modules such as `src/run_evidence_extraction.py` and `src/effect_intensity.py` for workflow entry points and shared domain logic. Research data is organized under `data/` by stage: `raw/`, `external/`, `interim/`, and `processed/`. Analysis notebooks live in `notebooks/`, and generated figures belong in `reports/figures/`.

## Build, Test, and Development Commands
Install dependencies with `uv sync`; use `pip install -r requirements.txt` only if `uv` is unavailable. Run quality checks with `uv run pre-commit run --all-files`. Format and lint Python directly with `uv run ruff format src` and `uv run ruff check src`. Typical workflows include `uv run python src/data/download.py` to refresh the paper list and `uv run python src/run_evidence_extraction.py` to process selected studies. For notebook work, start Jupyter with `uv run jupyter lab`.

## Coding Style & Naming Conventions
Target Python 3.10+ and follow Ruff’s configured style: 4-space indentation, double quotes, and a 120-character line limit. Keep imports sorted and grouped automatically by Ruff. Prefer `snake_case` for modules, functions, variables, and data files; use descriptive directory names for paper-specific folders such as `data/processed/<paperkey>/`. Keep scripts focused on one workflow step and move reusable logic into `src/.../utils.py` modules.

## Testing Guidelines
There is no dedicated automated test suite yet, so contributors should treat linting and reproducible reruns as the baseline quality gate. Before opening a PR, run `uv run pre-commit run --all-files` and rerun the relevant pipeline or notebook cells that your change affects. When adding tests, place them in a new `tests/` package and name files `test_<module>.py`.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit messages such as `Add two more papers from snowballing` and `Update forest plot figures and notebook version`. Keep commits narrowly scoped and explain the dataset, notebook, or script touched. PRs should include a concise summary, note any regenerated outputs under `data/processed/` or `reports/figures/`, and link the related issue or study task. Add screenshots only when figure output or notebook visuals materially changed.

## Data & Configuration Notes
Copy `dot-env-template` to `.env` when working with LLM-assisted selection and keep secrets out of version control. Large raw source files may trip the pre-commit size check, so avoid committing new binaries unless they are essential and reviewed.

## Agent skills

### Issue tracker

Issues live as markdown files under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles mapped to `Status:` lines in issue files. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
