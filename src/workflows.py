from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys

from google import genai
import polars as pl

from src.config import FIGURES_DIR, INTERIM_DATA_DIR, ROOT_DIR, TABLES_DIR, processed_paper_path
from src.data.download import clean_titles, download_arxiv_papers, papers_dict_to_polars_df
from src.data.papers.entities import Paper, Papers
from src.data.selection.select_papers import (
    PAPERS_FILENAME,
    SAMPLE_PAPERS_FILENAME,
    SCORES_FILENAME,
    collect_query_results,
    load_selection_papers,
    scores_to_frame,
    write_scores,
)
from src.data.utils import read_scopus_quantization_papers
from src.run_evidence_extraction import extract_knowledge_from
from src.tables.studies_summary import generate_studies_summary_tables
from src.validated_synthesis import write_all_validated_outputs

NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
EVIDENCE_ANALYSIS_NOTEBOOK = NOTEBOOKS_DIR / "5.0-evidence-analysis.ipynb"
DEFAULT_DOWNLOAD_MAX_RESULTS = 1000
DEFAULT_DOWNLOAD_QUERY = (
    '(ti:("machine learning" OR ML OR "deep learning" OR DL OR "large language model?" OR "LLM?" OR '
    '"neural network?" OR "?NN?" OR "f?undational model?" OR agent) AND (quantization OR quantize OR '
    'quantized) AND ("energy consumption" OR "energy efficien*" OR "sustain*" OR "carbon footprint" OR '
    '"carbon emission") ANDNOT ("FL" OR "federated learning")) OR (abs:("machine learning" OR ML OR '
    '"deep learning" OR DL OR "large language model?" OR "LLM?" OR "neural network?" OR "?NN?" OR '
    '"f?undational model?" OR agent) AND (quantization OR quantize OR quantized) AND ("energy '
    'consumption" OR "energy efficien*" OR "sustain*" OR "carbon footprint" OR "carbon emission") '
    'ANDNOT ("FL" OR "federated learning")) AND submittedDate:[202201010000 TO 202502040000]'
)
CORE_FIGURES = (
    FIGURES_DIR / "metrics-usage-distribution.pdf",
    FIGURES_DIR / "correctness-forestplot.pdf",
    FIGURES_DIR / "resource-efficiency-forestplot.pdf",
    FIGURES_DIR / "performance-forestplot.pdf",
)
CORE_TABLES = (
    TABLES_DIR / "studies-quasi-experiments.tex",
    TABLES_DIR / "studies-observational.tex",
    TABLES_DIR / "aggregated-effects.tex",
    TABLES_DIR / "belief-assignment.tex",
    TABLES_DIR / "leave-one-study-out.tex",
    TABLES_DIR / "sensitivity-mass-preserving.tex",
    TABLES_DIR / "subgroup-ptq-w-int8-a-int8.tex",
    TABLES_DIR / "intensity-thresholds.tex",
    TABLES_DIR / "result-macros.tex",
)
PROCESSED_OUTPUT_FILENAMES = (
    "improvement_metrics.parquet",
    "improvement_statistics_by_configuration.parquet",
    "improvement_statistics_by_precision.parquet",
    "effects_by_configuration.json",
    "effects_by_precision.json",
)
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"


@dataclass(frozen=True)
class ExternalDataStatus:
    paper_key: str
    status: str
    message: str


def get_selected_papers(paper_keys: list[str] | None = None) -> list[Paper]:
    papers_by_key = {paper.value.KEY: paper.value for paper in Papers}
    if paper_keys is None:
        return list(papers_by_key.values())

    unknown_keys = sorted(set(paper_keys) - set(papers_by_key))
    if unknown_keys:
        available = ", ".join(sorted(papers_by_key))
        requested = ", ".join(unknown_keys)
        raise ValueError(f"Unknown paper key(s): {requested}. Available keys: {available}")

    return [papers_by_key[key] for key in paper_keys]


def list_paper_keys() -> list[str]:
    return sorted(paper.value.KEY for paper in Papers)


def _missing_external_members(paper: Paper) -> list[str]:
    if paper.REMOTE_ARCHIVE_SOURCE is not None:
        return [
            member.local_filename
            for member in paper.REMOTE_ARCHIVE_SOURCE.members
            if not Path(paper.external_data_path(member.local_filename)).is_file()
        ]

    default_path = Path(paper.external_data_path())
    return [] if default_path.is_file() else [default_path.name]


def ensure_external_data(
    paper_keys: list[str] | None = None,
    *,
    download_missing: bool = False,
) -> list[ExternalDataStatus]:
    statuses: list[ExternalDataStatus] = []

    for paper in get_selected_papers(paper_keys):
        missing_files = _missing_external_members(paper)
        if not missing_files:
            statuses.append(ExternalDataStatus(paper.KEY, "present", "All required external data is present."))
            continue

        if paper.REMOTE_ARCHIVE_SOURCE is not None and download_missing:
            paper.ensure_external_data()
            statuses.append(
                ExternalDataStatus(
                    paper.KEY,
                    "downloaded",
                    f"Downloaded missing external data: {', '.join(missing_files)}.",
                )
            )
            continue

        if paper.REMOTE_ARCHIVE_SOURCE is not None:
            statuses.append(
                ExternalDataStatus(
                    paper.KEY,
                    "missing-downloadable",
                    "Missing downloadable external data: "
                    f"{', '.join(missing_files)}. Re-run with --download-missing to fetch it.",
                )
            )
            continue

        readme_path = Path(paper.external_data_path("README.md"))
        statuses.append(
            ExternalDataStatus(
                paper.KEY,
                "missing-manual",
                f"Missing manually supplied external data: {', '.join(missing_files)}. See {readme_path}.",
            )
        )

    return statuses


def validate_external_data_ready(statuses: list[ExternalDataStatus]) -> None:
    blocking = [status for status in statuses if status.status.startswith("missing")]
    if not blocking:
        return

    details = "\n".join(f"- {status.paper_key}: {status.message}" for status in blocking)
    raise RuntimeError(f"External data is not ready for all requested papers:\n{details}")


def download_paper_catalog(
    query: str = DEFAULT_DOWNLOAD_QUERY,
    *,
    max_results: int = DEFAULT_DOWNLOAD_MAX_RESULTS,
    output_path: Path = INTERIM_DATA_DIR / PAPERS_FILENAME,
) -> Path:
    arxiv_papers = download_arxiv_papers(query, max_results)
    arxiv_df = papers_dict_to_polars_df(arxiv_papers)
    scopus_df = read_scopus_quantization_papers().sort("Year", descending=False)

    papers = (
        clean_titles(pl.concat([scopus_df, arxiv_df], how="diagonal_relaxed"))
        .with_columns(pl.col("Title").str.to_lowercase().alias("Temp Title"))
        .unique("Temp Title", keep="first", maintain_order=True)
        .drop("Temp Title")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    papers.write_csv(output_path)
    return output_path


def validate_gemini_api_key() -> None:
    if os.getenv(GEMINI_API_KEY_ENV_VAR):
        return
    raise RuntimeError(
        f"Missing {GEMINI_API_KEY_ENV_VAR}. Set it in your environment or .env before running LLM selection."
    )


def run_llm_selection(
    *,
    papers_path: Path = INTERIM_DATA_DIR / PAPERS_FILENAME,
    sample_papers_path: Path = INTERIM_DATA_DIR / SAMPLE_PAPERS_FILENAME,
    output_path: Path = INTERIM_DATA_DIR / SCORES_FILENAME,
) -> Path:
    validate_gemini_api_key()

    relevant_data = load_selection_papers(papers_path=papers_path, sample_papers_path=sample_papers_path)
    client = genai.Client()
    scores_df = scores_to_frame(collect_query_results(client, relevant_data))
    write_scores(scores_df, output_path=output_path)
    return output_path


def run_evidence_extraction_workflow(
    paper_keys: list[str] | None = None,
    *,
    download_missing: bool = False,
) -> list[Path]:
    statuses = ensure_external_data(paper_keys, download_missing=download_missing)
    validate_external_data_ready(statuses)

    output_paths: list[Path] = []
    for paper in get_selected_papers(paper_keys):
        extract_knowledge_from(paper)
        for filename in PROCESSED_OUTPUT_FILENAMES:
            output_paths.append(processed_paper_path(paper.KEY, filename))

    validate_paths_exist(output_paths, label="processed outputs")
    return output_paths


def run_notebook_headless(notebook_path: Path = EVIDENCE_ANALYSIS_NOTEBOOK) -> Path:
    notebook_path = Path(notebook_path)
    executed_notebook = notebook_path.stem + ".executed.ipynb"
    output_dir = ROOT_DIR / ".tmp" / "executed-notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--output",
            executed_notebook,
            "--output-dir",
            str(output_dir),
            str(notebook_path),
        ],
        check=True,
    )

    executed_path = output_dir / executed_notebook
    validate_paths_exist([executed_path], label="executed notebook")
    return executed_path


def reproduce_figures(*, run_notebook: bool = True) -> list[Path]:
    if run_notebook:
        run_notebook_headless()
    validate_paths_exist(CORE_FIGURES, label="core figure outputs")
    return list(CORE_FIGURES)


def reproduce_tables() -> list[Path]:
    paths = list(generate_studies_summary_tables(output_dir=TABLES_DIR))
    paths.extend(write_all_validated_outputs())
    validate_paths_exist(CORE_TABLES, label="core table outputs")
    return paths


def reproduce_full_pipeline(*, download_missing: bool = False, run_notebook: bool = True) -> list[Path]:
    output_paths = run_evidence_extraction_workflow(download_missing=download_missing)
    if run_notebook:
        output_paths.extend(reproduce_figures(run_notebook=True))
    return output_paths


def validate_paths_exist(paths: list[Path] | tuple[Path, ...], *, label: str) -> None:
    missing = [Path(path) for path in paths if not Path(path).exists()]
    if not missing:
        return

    details = "\n".join(f"- {path}" for path in missing)
    raise RuntimeError(f"Missing expected {label}:\n{details}")
