# AGENTS.md

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
