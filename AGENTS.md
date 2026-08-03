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

## Cursor Cloud specific instructions

This repo is the single-product replication package for an academic paper on model quantization
(a Python data/analysis pipeline, not a web app). The primary interface is **Jupyter Lab** running
the notebooks in `notebooks/`. See `README.md` for the canonical usage/workflow overview.

### Environment essentials (already handled by the startup/update script)
- Managed with **uv** (`uv.lock` + `pyproject.toml`), Python 3.12. Run tooling via `uv run ...`.
- Notebooks and most data files (`*.ipynb`, `*.csv`, `*.parquet`, `*.xlsx`) are stored in **Git LFS**.
  They must be materialized with `git lfs pull` or they will only be small pointer files (~130 bytes).

### `.env` / config (non-obvious, required)
- `src/config.py` calls `load_dotenv()` and raises `ValueError` if `ROOT` is unset, so a `.env` file
  (git-ignored) with `ROOT=/workspace` is required. The update script recreates it if missing.
- **`PYTHONPATH` from `.env` is NOT enough for notebooks.** `load_dotenv()` runs after the Python
  process starts, so it cannot affect `sys.path`. To import the `src` package from a notebook kernel
  or from `jupyter nbconvert`, you must export `PYTHONPATH=/workspace` as a real env var *before*
  launching Jupyter, otherwise cells fail with `ModuleNotFoundError: No module named 'src'`.

### Running the app (Jupyter Lab)
- Start from the repo root with `PYTHONPATH` exported, e.g.:
  `PYTHONPATH=/workspace uv run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.token="" --ServerApp.password=""`
- Notebooks `3.0`, `4.0`, `5.0` run fully **offline** against committed processed data
  (`data/processed/`). `5.0-evidence-analysis.ipynb` is the best end-to-end check: it regenerates
  the forest plots into `reports/figures/`.
- Notebooks `1.0` and `2.0` call the **Gemini API** and require a real `GEMINI_API_KEY` in `.env`
  (only needed to regenerate LLM paper selection; not needed for the analysis/plots).

### Lint / test / build
- Lint: `uv run ruff check .`, `uv run ruff format --check .`, `uv run deptry .`. Note the repo
  currently has pre-existing ruff/deptry findings and two notebooks that are not ruff-formatted;
  these are upstream, not environment problems.
- There is **no automated test suite** and no CI config. "Testing" means executing the notebooks,
  e.g. `PYTHONPATH=/workspace uv run jupyter nbconvert --to notebook --execute notebooks/5.0-evidence-analysis.ipynb --output /tmp/out.ipynb`.
- There is no local build step; the `Dockerfile` builds the published image and installs
  `ttf-mscorefonts-installer` only for figure fonts.

### Known cosmetic caveat
- Plots emit `findfont: ... 'Times New Roman' not found` warnings unless MS core fonts are installed
  (the Dockerfile does this). Matplotlib falls back to a default serif font; figures still render
  correctly, so this is safe to ignore in the dev environment.

### Git LFS / data hygiene
- After `git lfs pull`, `git status` may show `data/external/*.csv` as "modified" with identical LFS
  OIDs (`LFS: <hash> -> File: <hash>`); this is a harmless pointer/working-tree artifact, not a real
  change. Running the analysis notebooks also rewrites the LFS-tracked `reports/figures/*` outputs.
  Do not commit these unless intentionally updating figures/data.
