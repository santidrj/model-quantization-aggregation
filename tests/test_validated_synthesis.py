"""Validated synthesis file: one source for manuscript numbers."""

import polars as pl

from src.belief_assignment import PUBLISHED_TABLE, leave_one_study_out_records, manuscript_comparison_records
from src.dempster_shafer import format_intensity
from src.validated_synthesis import (
    assert_validated_synthesis,
    build_validated_synthesis,
    forestplot_overlay_frame,
    render_aggregated_effects_table,
    render_result_macros,
    validate_synthesis,
    write_all_validated_outputs,
)


def test_validated_synthesis_assertions_pass():
    payload = build_validated_synthesis()
    assert validate_synthesis(payload) == []
    assert_validated_synthesis(payload)


def test_main_table_effects_match_published_gate():
    payload = build_validated_synthesis()
    for expected in PUBLISHED_TABLE:
        row = payload["effects"][expected.effect]
        assert row["intensity_label"] == format_intensity(expected.intensity)
        assert row["belief_percent"] == expected.belief_percent
        assert row["n_primary_studies"] == expected.n_primary_studies
        assert row["n_evidence_models"] == expected.n_evidence_models


def test_appendix_analogue_matches_main_table():
    payload = build_validated_synthesis()
    records = {record["effect"]: record for record in manuscript_comparison_records()}
    for expected in PUBLISHED_TABLE:
        analogue = payload["belief_assignment"]["effects"][expected.effect]
        main = payload["effects"][expected.effect]
        record = records[expected.effect]
        assert analogue["analogue_intensity"] == main["intensity_label"]
        assert analogue["analogue_belief_percent"] == main["belief_percent"]
        assert record["analogue_intensity"] == format_intensity(expected.intensity)
        assert record["analogue_belief_percent"] == expected.belief_percent


def test_gpu_energy_leave_one_out_reports_all_omissions():
    rows = [row for row in leave_one_study_out_records() if row.effect == "GPU Energy Consumption"]
    assert {row.omitted_study_id for row in rows} == {"S3", "S13", "S14"}
    assert all(row.belief_delta > 0 for row in rows)


def test_forestplot_overlay_uses_main_gpu_utilization():
    payload = build_validated_synthesis()
    overlay = payload["forestplot_overlays"]["GPU Utilization"]
    main = payload["effects"]["GPU Utilization"]
    assert overlay["belief_percent"] == main["belief_percent"]
    assert overlay["intensity"] == main["intensity"]
    frame = forestplot_overlay_frame(payload)
    gpu = frame.filter(pl.col("effect") == "GPU Utilization").to_dicts()[0]
    assert gpu["belief"] == round(main["belief"], 3)
    assert gpu["lower_ci"] == int(overlay["forestplot"]["lower_ci"])
    assert gpu["upper_ci"] == int(overlay["forestplot"]["upper_ci"])


def test_counts_ten_increased_one_decreased():
    payload = build_validated_synthesis()
    assert payload["counts"]["n_increased_belief"] == len(payload["counts"]["increased_effects"])
    assert payload["counts"]["n_decreased_belief"] == 1  # noqa: PLR2004
    assert payload["counts"]["decreased_effects"] == ["RAM Usage"]
    assert "F1 Score" in payload["counts"]["increased_effects"]
    assert "Accuracy" in payload["counts"]["increased_effects"]
    assert "GPU Utilization" in payload["counts"]["increased_effects"]
    assert payload["counts"]["n_increased_belief"] == 10  # noqa: PLR2004


def test_subgroup_has_nine_studies_and_complete_source_list():
    payload = build_validated_synthesis()
    subgroup = payload["subgroup"]
    assert subgroup["n_primary_studies"] == len(subgroup["study_ids"])
    assert subgroup["n_evidence_models"] == len(subgroup["study_ids"])
    assert subgroup["one_model_per_study"] is True
    assert subgroup["study_ids"] == ["S3", "S4", "S6", "S10", "S15", "S17", "S18", "S20", "S21"]
    latency_n = subgroup["effects"]["Inference Latency"]["n_primary_studies"]
    storage_n = subgroup["effects"]["Storage Size"]["n_evidence_models"]
    assert latency_n == storage_n
    assert latency_n == len(subgroup["effects"]["Inference Latency"]["study_ids"])


def test_aggregated_effects_table_uses_validated_beliefs():
    latex = render_aggregated_effects_table(build_validated_synthesis())
    assert r"GPU utilization & \{IF, WP\} & \textbf{97\%}" in latex
    assert "Inf. power draw" in latex and "74\\%" in latex
    assert "Inf. latency" in latex and "100\\%" in latex
    assert "F$_1$-score" in latex


def test_write_all_validated_outputs(tmp_path):
    paths = write_all_validated_outputs(json_path=tmp_path / "validated-synthesis.json", output_dir=tmp_path)
    names = {path.name for path in paths}
    assert "validated-synthesis.json" in names
    assert "aggregated-effects.tex" in names
    assert "intensity-thresholds.tex" in names
    assert "intensity-threshold-sensitivity.tex" in names
    assert "leave-one-study-out.tex" in names
    loo = (tmp_path / "leave-one-study-out.tex").read_text(encoding="utf-8")
    gpu_energy_rows = [row for row in loo.splitlines() if row.startswith("GPU energy")]
    assert len(gpu_energy_rows) == len({"S3", "S13", "S14"})
    analogue = (tmp_path / "belief-assignment.tex").read_text(encoding="utf-8")
    assert "mAP & 2 & 4 & IF & 0.45" in analogue
    assert "RAM usage & 2 & 3 & SP & 0.47" in analogue
    assert "Accuracy & 10 & 41 & IF & 0.91" in analogue


def test_intensity_reconciliation_intersects_primary_and_mass_preserving():
    payload = build_validated_synthesis()
    recon = payload["intensity_reconciliation"]["effects"]
    assert recon["Accuracy"]["intensity"] == ["IF"]
    assert recon["Accuracy"]["differs_from_primary"] is False
    assert recon["Accuracy"]["has_theory_arrow"] is True
    assert recon["Inference Latency"]["intensity"] == ["SP"]
    assert recon["Inference Latency"]["differs_from_primary"] is True
    assert recon["RAM Usage"]["intensity"] is None
    assert recon["RAM Usage"]["has_theory_arrow"] is False
    assert recon["Storage Size"]["differs_from_primary"] is False
    macros = render_result_macros(payload)
    assert r"\newcommand{\TheoryAccuracyIntensity}{IF}" in macros
    assert r"\newcommand{\TheoryAccuracyProse}{indifferently}" in macros
    assert r"\newcommand{\TheoryInfLatencyIntensity}{SP}" in macros
    assert r"\newcommand{\TheoryRAMUsageHasArrow}{false}" in macros
