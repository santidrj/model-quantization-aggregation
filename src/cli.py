from __future__ import annotations

from pathlib import Path

import click

from src import workflows


def _print_paths(paths: tuple[Path, ...] | list[Path], label: str) -> None:
    click.echo(f"{label}:")
    for path in paths:
        click.echo(f"- {path}")


@click.group(help="Reproduce deterministic outputs and maintain the paper-processing workflow.")
def main() -> None:
    """Top-level CLI group."""


@main.group(help="Deterministic reproduction workflows.")
def reproduce() -> None:
    """Deterministic reproduction workflows."""


@reproduce.command("figures", help="Regenerate the core paper figures from processed data.")
@click.option(
    "--no-notebooks",
    is_flag=True,
    help="Skip notebook execution and only validate that expected figure files already exist.",
)
def reproduce_figures(no_notebooks: bool) -> None:
    paths = workflows.reproduce_figures(run_notebook=not no_notebooks)
    _print_paths(paths, "Validated figure outputs")


@reproduce.command("tables", help="Regenerate manuscript studies-summary LaTeX tabular fragments.")
def reproduce_tables() -> None:
    paths = workflows.reproduce_tables()
    _print_paths(paths, "Validated table outputs")


@reproduce.command("full-pipeline", help="Regenerate processed evidence and optionally the core figures.")
@click.option(
    "--download-missing",
    is_flag=True,
    help="Allow auto-download for papers that support fetching missing external data.",
)
@click.option(
    "--no-notebooks",
    is_flag=True,
    help="Skip notebook execution and stop after validating processed outputs.",
)
def reproduce_full_pipeline(download_missing: bool, no_notebooks: bool) -> None:
    paths = workflows.reproduce_full_pipeline(
        download_missing=download_missing,
        run_notebook=not no_notebooks,
    )
    _print_paths(paths, "Validated reproduction outputs")


@main.group(help="Paper maintenance workflows.")
def papers() -> None:
    """Paper maintenance workflows."""


@papers.command("list", help="List supported paper keys.")
def list_papers() -> None:
    for key in workflows.list_paper_keys():
        click.echo(key)


@papers.command("download", help="Refresh the candidate paper catalog from Scopus and arXiv.")
def download_papers() -> None:
    output_path = workflows.download_paper_catalog()
    click.echo(f"Wrote refreshed paper catalog to {output_path}")


@papers.command("select", help="Run the non-deterministic LLM paper-selection step.")
@click.option(
    "--run-llm",
    is_flag=True,
    help="Required acknowledgement that this command performs live LLM calls.",
)
def select_papers(run_llm: bool) -> None:
    if not run_llm:
        raise click.UsageError("`mq papers select` requires --run-llm because it performs live LLM calls.")
    output_path = workflows.run_llm_selection()
    click.echo(f"Wrote LLM selection scores to {output_path}")


@papers.command(
    "build-selection-manifest",
    help="Build the record-level selection manifest from frozen screening artifacts.",
)
def build_selection_manifest() -> None:
    paths = workflows.build_selection_manifest_outputs()
    _print_paths(paths, "Selection manifest outputs")


@papers.command("ensure-external-data", help="Preflight or fetch external paper data for one or more papers.")
@click.option(
    "--paper",
    "paper_keys",
    multiple=True,
    help="Paper key to target. Repeat to select multiple papers.",
)
@click.option(
    "--download-missing",
    is_flag=True,
    help="Allow auto-download for papers that support fetching missing external data.",
)
def ensure_external_data(paper_keys: tuple[str, ...], download_missing: bool) -> None:
    statuses = workflows.ensure_external_data(list(paper_keys) or None, download_missing=download_missing)
    for status in statuses:
        click.echo(f"{status.paper_key}: {status.status} - {status.message}")
    workflows.validate_external_data_ready(statuses)


@main.group(help="Lower-level evidence extraction workflows.")
def extraction() -> None:
    """Lower-level evidence extraction workflows."""


@extraction.command("run", help="Run evidence extraction for all or selected papers.")
@click.option(
    "--paper",
    "paper_keys",
    multiple=True,
    help="Paper key to target. Repeat to select multiple papers.",
)
@click.option(
    "--download-missing",
    is_flag=True,
    help="Allow auto-download for papers that support fetching missing external data.",
)
def run_extraction(paper_keys: tuple[str, ...], download_missing: bool) -> None:
    paths = workflows.run_evidence_extraction_workflow(
        paper_keys=list(paper_keys) or None,
        download_missing=download_missing,
    )
    _print_paths(paths, "Validated processed outputs")


if __name__ == "__main__":
    main()
