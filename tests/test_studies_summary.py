"""Studies-summary LaTeX table generation (manuscript fragments)."""

from pathlib import Path

from click.testing import CliRunner
import pytest

from src import cli
from src.tables.studies_summary import (
    OBSERVATIONAL,
    QUASI_EXPERIMENTS,
    StudySummaryRow,
    energy_evidence_group,
    format_belief_percent,
    format_instrumentation,
    format_models_data,
    format_source_precision,
    format_target_precision,
    format_timing,
    paper_key_author_macro,
    render_tabularx_body,
    study_id_latex,
    write_studies_summary_tables,
)


@pytest.mark.parametrize(
    ("methods", "expected"),
    [
        (None, "quasi-experiments"),
        ([], "quasi-experiments"),
        (["hardware-based"], "quasi-experiments"),
        (["software-based"], "quasi-experiments"),
        (["hardware-based", "software-based"], "quasi-experiments"),
        (["analytical"], "observational"),
        (["model-based"], "observational"),
        (["analytical", "model-based"], "observational"),
    ],
)
def test_energy_evidence_group(methods, expected):
    assert energy_evidence_group(methods) == expected


def test_energy_evidence_group_rejects_mixed_measured_and_estimated():
    with pytest.raises(ValueError, match="energy evidence group"):
        energy_evidence_group(["hardware-based", "analytical"])


@pytest.mark.parametrize(
    ("paper_key", "expected"),
    [
        ("gonzalezImpactMLOptimization2025", "gonzalez"),
        ("deputterPOQThereParetoOptimal2025", "deputter"),
        ("alizadehLanguageModelsSoftware2025", "alizadeh"),
    ],
)
def test_paper_key_author_macro(paper_key, expected):
    assert paper_key_author_macro(paper_key) == expected


def test_study_id_latex():
    assert study_id_latex("gonzalezImpactMLOptimization2025") == (r"\gonzalez \cite{gonzalezImpactMLOptimization2025}")


def test_format_belief_percent():
    assert format_belief_percent(0.7361083333333334) == "74\\%"


@pytest.mark.parametrize(
    ("methods", "expected"),
    [
        (["ptq"], "PTQ"),
        (["qat"], "QAT"),
        (["ptq-retrain"], "PTQ-R"),
        (["qat", "ptq"], "QAT, PTQ"),
        (["ptq", "qat"], "PTQ, QAT"),
    ],
)
def test_format_timing(methods, expected):
    assert format_timing(methods) == expected


def test_format_models_data():
    assert format_models_data(["a", "b"], ["x"]) == "2 / 1"


@pytest.mark.parametrize(
    ("methods", "software_tools", "expected"),
    [
        (None, None, "N/A"),
        ([], [], "N/A"),
        (["analytical"], [], "Analytical"),
        (["model-based"], [], "Model-based"),
        (["hardware-based"], [], "Hardware-based"),
        (["software-based"], ["nvidia-smi"], "nvidia-smi"),
        (["software-based"], ["pyNVML", "pyRAPL"], "pyNVML, pyRAPL"),
        (["software-based", "hardware-based"], ["nvidia-smi"], "Hardware + nvidia-smi"),
    ],
)
def test_format_instrumentation(methods, software_tools, expected):
    assert format_instrumentation(methods, software_tools) == expected


@pytest.mark.parametrize(
    ("baseline", "expected"),
    [
        ("full-fp32", "FP32"),
        ("full-fp16", "FP16"),
        ("full-fp64", "FP64"),
    ],
)
def test_format_source_precision(baseline, expected):
    assert format_source_precision(baseline) == expected


@pytest.mark.parametrize(
    ("configs", "baseline", "expected"),
    [
        (["w-int8, a-int8"], "full-fp32", "INT8 (WA)"),
        (["w-int4", "w-int8"], "full-fp32", "INT4, INT8 (W)"),
        (["w-int8, a-int8, b-int8"], "full-fp32", "INT8 (WAB)"),
        (["full-int8"], "full-fp32", "INT8 (F)"),
        (["w-int8, a-int8", "full-int8"], "full-fp32", "INT8 (WA, F)"),
        (["mixed"], "full-fp16", "Mixed (WA)"),
        (
            [
                "w-q0.8, a-q0.8",
                "w-q0.8, a-fp32",
                "w-fp32, a-q0.8",
                "w-q0.16, a-q0.16",
                "w-q0.16, a-fp32",
                "w-fp32, a-q0.16",
                "w-q0.8, a-q0.16",
                "w-q0.16, a-q0.8",
            ],
            "full-fp32",
            "Q0.8, Q0.16 (W, A, WA)",
        ),
        (
            ["w-q0.4, a-q0.4", "w-q0.8, a-q0.8", "w-q0.16, a-q0.16", "w-q0.32, a-q0.32"],
            "full-fp32",
            "Q0.4--Q0.32 (WA)",
        ),
        (["w-int8, a-int8", "w-fp16, a-fp16"], "full-fp32", "INT8, FP16 (WA)"),
    ],
)
def test_format_target_precision(configs, baseline, expected):
    assert format_target_precision(configs, baseline) == expected


def test_render_tabularx_body_contains_header_and_row():
    rows = [
        StudySummaryRow(
            paper_key="koliEdgeAIPoweredSystem2025",
            study_id="S19",
            belief=0.6533320833333334,
            domain="Plant disease",
            instrumentation="N/A",
            models_data="2 / 1",
            source="FP32",
            target="INT8 (W)",
            timing="PTQ",
            ts=1,
        )
    ]
    latex = render_tabularx_body(rows)
    assert r"\begin{tabularx}" in latex
    assert r"\koli \cite{koliEdgeAIPoweredSystem2025}" in latex
    assert "Plant disease" in latex
    assert "65\\%" in latex
    assert r"\end{tabularx}" in latex


def test_write_studies_summary_tables_splits_groups(tmp_path: Path):
    rows = [
        StudySummaryRow(
            paper_key="taoExperimentalEnergyConsumption2022",
            study_id="S11",
            belief=0.64,
            domain="Bird call",
            instrumentation="Hardware-based",
            models_data="1 / 1",
            source="FP32",
            target="Q0.8, Q0.16 (W, A, WA)",
            timing="QAT",
            ts=8,
            energy_evidence_group=QUASI_EXPERIMENTS,
        ),
        StudySummaryRow(
            paper_key="vasquezActivationDensityBased2021",
            study_id="S5",
            belief=0.38,
            domain="Image class.",
            instrumentation="Model-based",
            models_data="2 / 2",
            source="FP16",
            target="Mixed (WA)",
            timing="QAT",
            ts=1,
            energy_evidence_group=OBSERVATIONAL,
        ),
        StudySummaryRow(
            paper_key="barnellModelQuantizationSynthetic2021",
            study_id="S3",
            belief=0.67,
            domain="Obj. detect.",
            instrumentation="nvidia-smi",
            models_data="1 / 1",
            source="FP32",
            target="INT8, FP16 (WA)",
            timing="PTQ",
            ts=2,
            energy_evidence_group=QUASI_EXPERIMENTS,
        ),
    ]
    paths = write_studies_summary_tables(rows, output_dir=tmp_path)
    assert paths == [
        tmp_path / "studies-quasi-experiments.tex",
        tmp_path / "studies-observational.tex",
    ]
    quasi = paths[0].read_text()
    observational = paths[1].read_text()
    assert r"\barnell" in quasi
    assert r"\tao" in quasi
    # Belief descending: barnell (67) before tao (64)
    assert quasi.index(r"\barnell") < quasi.index(r"\tao")
    assert r"\vasquez" in observational
    assert r"\barnell" not in observational


def test_format_target_precision_log_formats():
    assert (
        format_target_precision(
            ["w-log4", "w-log3", "w-log2", "w-log1"],
            "full-fp32",
        )
        == "LOG1--LOG4 (W)"
    )


def test_reproduce_tables_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    written = [tmp_path / "studies-quasi-experiments.tex", tmp_path / "studies-observational.tex"]
    for path in written:
        path.write_text("% stub\n", encoding="utf-8")

    monkeypatch.setattr(cli.workflows, "reproduce_tables", lambda: written)

    runner = CliRunner()
    result = runner.invoke(cli.main, ["reproduce", "tables"])

    assert result.exit_code == 0
    assert "Validated table outputs" in result.output


def test_belief_percent_ties_break_by_study_id():
    rows = [
        StudySummaryRow(
            paper_key="deputterPOQThereParetoOptimal2025",
            study_id="S16",
            belief=0.4541629166666667,
            domain="Image class.",
            instrumentation="Analytical",
            models_data="2 / 1",
            source="FP16",
            target="INT2--INT8 (WA, F)",
            timing="QAT",
            ts=8,
            energy_evidence_group=OBSERVATIONAL,
        ),
        StudySummaryRow(
            paper_key="denkingerImpactMemoryVoltage2020",
            study_id="S2",
            belief=0.44707958333333336,
            domain="Coffee rec.",
            instrumentation="Analytical",
            models_data="1 / 1",
            source="FP32",
            target="Q0.4--Q0.32 (WA)",
            timing="QAT",
            ts=3,
            energy_evidence_group=OBSERVATIONAL,
        ),
    ]
    latex = render_tabularx_body(rows)
    assert latex.index(r"\denkinger") < latex.index(r"\deputter")
