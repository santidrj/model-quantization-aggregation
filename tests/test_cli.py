from pathlib import Path

from click.testing import CliRunner
import pytest

from src import cli

CLI_USAGE_ERROR = 2


def test_papers_select_requires_run_llm() -> None:
    runner = CliRunner()

    result = runner.invoke(cli.main, ["papers", "select"])

    assert result.exit_code == CLI_USAGE_ERROR
    assert "--run-llm" in result.output


def test_papers_list_prints_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.workflows, "list_paper_keys", lambda: ["paper-a", "paper-b"])

    runner = CliRunner()
    result = runner.invoke(cli.main, ["papers", "list"])

    assert result.exit_code == 0
    assert result.output == "paper-a\npaper-b\n"


def test_reproduce_full_pipeline_passes_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_reproduce_full_pipeline(*, download_missing: bool, run_notebook: bool) -> list[Path]:
        captured["download_missing"] = download_missing
        captured["run_notebook"] = run_notebook
        return [Path("reports/figures/correctness-forestplot.pdf")]

    monkeypatch.setattr(cli.workflows, "reproduce_full_pipeline", fake_reproduce_full_pipeline)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["reproduce", "full-pipeline", "--download-missing", "--no-notebooks"])

    assert result.exit_code == 0
    assert captured == {"download_missing": True, "run_notebook": False}
    assert "Validated reproduction outputs" in result.output
