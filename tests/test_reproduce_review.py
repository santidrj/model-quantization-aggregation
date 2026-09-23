from pathlib import Path

import pytest

from src import workflows


def test_reproduce_notebook_selects_the_subgroup_notebook(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Path] = {}

    def fake_run(notebook_path: Path) -> Path:
        seen["path"] = notebook_path
        return notebook_path

    monkeypatch.setattr(workflows, "run_notebook_headless", fake_run)

    executed = workflows.reproduce_notebook("5.1")

    assert seen["path"].name == "5.1-subgroup-ptq-w-int8-a-int8.ipynb"
    assert executed.name == "5.1-subgroup-ptq-w-int8-a-int8.ipynb"


def test_reproduce_review_stops_when_external_paper_data_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fail_ready(_statuses: list[object]) -> None:
        calls.append("ensure-external-data")
        raise RuntimeError("External data is not ready for all requested papers")

    monkeypatch.setattr(workflows, "ensure_external_data", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(workflows, "validate_external_data_ready", fail_ready)
    monkeypatch.setattr(
        workflows,
        "run_evidence_extraction_workflow",
        lambda **_kwargs: calls.append("extraction") or [],
    )

    with pytest.raises(RuntimeError, match="External data is not ready"):
        workflows.reproduce_review()

    assert calls == ["ensure-external-data"]


def test_reproduce_review_runs_the_bar_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(workflows, "ensure_external_data", lambda *_args, **_kwargs: calls.append("ensure") or [])
    monkeypatch.setattr(workflows, "validate_external_data_ready", lambda _statuses: None)
    monkeypatch.setattr(
        workflows,
        "run_evidence_extraction_workflow",
        lambda **_kwargs: calls.append("extraction") or [Path("processed")],
    )
    monkeypatch.setattr(
        workflows,
        "reproduce_figures",
        lambda **_kwargs: calls.append("figures") or [Path("figure.pdf")],
    )
    monkeypatch.setattr(workflows, "reproduce_tables", lambda: calls.append("tables") or [Path("table.tex")])
    monkeypatch.setattr(
        workflows,
        "reproduce_notebook",
        lambda notebook_id: calls.append(f"notebook:{notebook_id}") or Path(f"{notebook_id}.ipynb"),
    )

    workflows.reproduce_review()

    assert calls == [
        "ensure",
        "extraction",
        "figures",
        "tables",
        "notebook:3.0",
        "notebook:4.0",
        "notebook:5.1",
        "notebook:6.0",
    ]
