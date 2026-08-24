"""Belief assignment: study-belief masses on evidence models."""

import json
from pathlib import Path

from src.belief_assignment import (
    PUBLISHED_TABLE,
    EvidenceModel,
    MassAssignment,
    assigned_mass,
    belief_assignment_trace,
    comparison_records,
    contributing_study_ids,
    leave_one_study_out_records,
    load_evidence_models,
    mass_preserving_sensitivity_rows,
    pieces_for_effect,
    render_belief_assignment_table,
    render_leave_one_study_out_table,
    render_sensitivity_mass_preserving_table,
    reproduction_mismatches,
    synthesis_row,
    write_belief_assignment_table,
    write_leave_one_study_out_table,
    write_sensitivity_mass_preserving_table,
)
from src.dempster_shafer import HypothesisSelectionPolicy, combine_effect

AJI_STUDY_BELIEF = 0.6720808333333333
AJI_EVIDENCE_MODEL_COUNT = 4
AJI_PROCESSED_BELIEF = 0.327
CORPUS_EVIDENCE_MODEL_COUNT = 76
ACCURACY_EVIDENCE_MODEL_COUNT = 41
ACCURACY_PRIMARY_STUDY_COUNT = 10
ACCURACY_LOO_FULL_BELIEF = 0.99
ACCURACY_LOO_BELIEF = 0.78
RAM_USAGE_EVIDENCE_MODEL_COUNT = 3
RAM_USAGE_SANTOS_BELIEF_PERCENT = 65
ROOT_EXAMPLE_STUDY_BELIEF = 0.75
ROOT_EXAMPLE_EVIDENCE_MODEL_COUNT = 2
ROOT_EXAMPLE_MASS = 0.5


def _aji_like_models() -> list[EvidenceModel]:
    return [
        EvidenceModel(
            study_id="S1",
            quantization_method="ptq-retrain",
            precision_configuration=f"w-log{index}",
            study_belief=AJI_STUDY_BELIEF,
            evidence_model_count=AJI_EVIDENCE_MODEL_COUNT,
            effects={
                "BLEU": ("indifferent", AJI_PROCESSED_BELIEF),
                "Storage Size": ("strongly positive", AJI_PROCESSED_BELIEF),
            },
        )
        for index in range(1, 5)
    ]


def test_split_assigns_study_belief_over_evidence_model_count():
    model = _aji_like_models()[0]
    assert assigned_mass(model, MassAssignment.EQUITABLE_BELIEF_SPLIT) == AJI_STUDY_BELIEF / AJI_EVIDENCE_MODEL_COUNT


def test_mass_preserving_split_assigns_simple_support_root():
    model = EvidenceModel(
        study_id="S1",
        quantization_method="ptq",
        precision_configuration="w-int8",
        study_belief=ROOT_EXAMPLE_STUDY_BELIEF,
        evidence_model_count=ROOT_EXAMPLE_EVIDENCE_MODEL_COUNT,
        effects={"Accuracy": ("positive", 0.4)},
    )
    assert assigned_mass(model, MassAssignment.MASS_PRESERVING_BELIEF_SPLIT) == ROOT_EXAMPLE_MASS


def test_mass_preserving_split_matches_unsplit_when_study_has_one_evidence_model():
    model = EvidenceModel(
        study_id="S1",
        quantization_method="ptq",
        precision_configuration="w-int8",
        study_belief=ROOT_EXAMPLE_STUDY_BELIEF,
        evidence_model_count=1,
        effects={"Accuracy": ("positive", 0.4)},
    )
    assert assigned_mass(model, MassAssignment.MASS_PRESERVING_BELIEF_SPLIT) == ROOT_EXAMPLE_STUDY_BELIEF
    assert assigned_mass(model, MassAssignment.UNDISCOUNTED_UNSPLIT) == ROOT_EXAMPLE_STUDY_BELIEF


def test_mass_preserving_split_uses_study_evidence_model_count_when_effect_is_missing_from_some_models():
    models = [
        EvidenceModel(
            study_id="S2",
            quantization_method="qat",
            precision_configuration="w-int8",
            study_belief=ROOT_EXAMPLE_STUDY_BELIEF,
            evidence_model_count=ROOT_EXAMPLE_EVIDENCE_MODEL_COUNT,
            effects={"Accuracy": ("indifferent", 0.4)},
        ),
        EvidenceModel(
            study_id="S2",
            quantization_method="qat",
            precision_configuration="w-int4",
            study_belief=ROOT_EXAMPLE_STUDY_BELIEF,
            evidence_model_count=ROOT_EXAMPLE_EVIDENCE_MODEL_COUNT,
            effects={"Accuracy": ("indifferent", 0.4), "RAM Usage": ("strongly positive", 0.5)},
        ),
    ]
    ram_pieces = pieces_for_effect(models, "RAM Usage", MassAssignment.MASS_PRESERVING_BELIEF_SPLIT)
    assert ram_pieces == [("strongly positive", ROOT_EXAMPLE_MASS)]


def test_mass_preserving_split_recovers_study_belief_when_all_models_agree():
    models = [
        EvidenceModel(
            study_id="S1",
            quantization_method="ptq",
            precision_configuration=f"w-int{8 if index == 0 else 4}",
            study_belief=ROOT_EXAMPLE_STUDY_BELIEF,
            evidence_model_count=ROOT_EXAMPLE_EVIDENCE_MODEL_COUNT,
            effects={"Accuracy": ("positive", 0.4)},
        )
        for index in range(2)
    ]
    pieces = pieces_for_effect(models, "Accuracy", MassAssignment.MASS_PRESERVING_BELIEF_SPLIT)
    combined = combine_effect(pieces)
    assert pieces == [("positive", ROOT_EXAMPLE_MASS), ("positive", ROOT_EXAMPLE_MASS)]
    assert combined.belief == ROOT_EXAMPLE_STUDY_BELIEF


def test_unsplit_assigns_full_study_belief():
    model = _aji_like_models()[0]
    assert assigned_mass(model, MassAssignment.UNDISCOUNTED_UNSPLIT) == AJI_STUDY_BELIEF


def test_published_analogue_uses_processed_belief():
    model = _aji_like_models()[0]
    assert assigned_mass(model, MassAssignment.PUBLISHED_ANALOGUE, effect="BLEU") == AJI_PROCESSED_BELIEF


def test_split_uses_study_evidence_model_count_when_effect_is_missing_from_some_models():
    models = [
        EvidenceModel(
            study_id="S2",
            quantization_method="qat",
            precision_configuration="w-int8",
            study_belief=0.6,
            evidence_model_count=2,
            effects={"Accuracy": ("indifferent", 0.4)},
        ),
        EvidenceModel(
            study_id="S2",
            quantization_method="qat",
            precision_configuration="w-int4",
            study_belief=0.6,
            evidence_model_count=2,
            effects={"Accuracy": ("indifferent", 0.4), "RAM Usage": ("strongly positive", 0.5)},
        ),
    ]
    ram_pieces = pieces_for_effect(models, "RAM Usage", MassAssignment.EQUITABLE_BELIEF_SPLIT)
    assert ram_pieces == [("strongly positive", 0.3)]


def test_load_evidence_models_assigns_study_level_n():
    models = load_evidence_models()
    aji = [model for model in models if model.study_id == "S1"]
    assert len(aji) == AJI_EVIDENCE_MODEL_COUNT
    assert {model.evidence_model_count for model in aji} == {AJI_EVIDENCE_MODEL_COUNT}
    assert len(models) == CORPUS_EVIDENCE_MODEL_COUNT
    assert [model.evidence_factory_id for model in aji] == [319366, 319396, 319424, 319452]


def test_aggregation_order_follows_evidence_factory_turn_list():
    models = load_evidence_models()
    ordered = sorted(models, key=lambda model: model.aggregation_index)
    identities = [(model.study_id, model.quantization_method, model.precision_configuration) for model in ordered]
    assert identities[0] == ("S1", "ptq-retrain", "w-log1")
    assert identities[1] == ("S6", "qat", "mixed-3.9")
    assert identities[-1] == ("S19", "ptq", "w-int8, a-fp32")
    assert len(set(identities)) == CORPUS_EVIDENCE_MODEL_COUNT
    assert [model.aggregation_index for model in ordered] == list(range(CORPUS_EVIDENCE_MODEL_COUNT))


def test_published_analogue_matches_intensity_and_model_counts():
    assert reproduction_mismatches(checks=("intensity", "n_primary_studies", "n_evidence_models")) == []


def test_published_analogue_matches_belief():
    assert reproduction_mismatches(checks=("belief",)) == []


def test_published_analogue_matches_conflict():
    assert reproduction_mismatches(checks=("conflict",)) == []


def test_synthesis_can_select_santos_2015_policy_explicitly():
    models = load_evidence_models()
    result = synthesis_row(
        models,
        "RAM Usage",
        MassAssignment.PUBLISHED_ANALOGUE,
        selection_policy=HypothesisSelectionPolicy.SANTOS_2015,
    )
    assert result.intensity == frozenset({"PO", "SP"})
    assert result.belief_percent == RAM_USAGE_SANTOS_BELIEF_PERCENT


def test_undiscounted_synthesis_uses_explicit_santos_policy():
    models = [
        EvidenceModel(
            study_id=f"S{index}",
            quantization_method="ptq",
            precision_configuration="w-int8",
            study_belief=mass,
            evidence_model_count=1,
            effects={"RAM Usage": (intensity, 0.1)},
            aggregation_index=index,
        )
        for index, (intensity, mass) in enumerate(
            [
                ("positive", 0.191),
                ("positive", 0.191),
                ("strongly positive", 0.572),
            ],
            start=1,
        )
    ]
    result = synthesis_row(
        models,
        "RAM Usage",
        MassAssignment.UNDISCOUNTED_UNSPLIT,
        selection_policy=HypothesisSelectionPolicy.SANTOS_2015,
    )
    assert result.intensity == frozenset({"PO", "SP"})


def test_comparison_records_use_santos_2015_for_every_local_assignment():
    records = {record["effect"]: record for record in comparison_records()}
    assert records["mAP"]["analogue_intensity"] == "{NE, WN, IF}"
    assert records["mAP"]["mass_preserving_intensity"] == "{NE, WN, IF}"
    assert records["mAP"]["equitable_intensity"] == "{NE, WN, IF}"
    assert records["RAM Usage"]["analogue_intensity"] == "{PO, SP}"
    assert records["RAM Usage"]["mass_preserving_intensity"] == "{PO, SP}"
    assert records["RAM Usage"]["equitable_intensity"] == "{PO, SP}"
    assert {record["selection_policy"] for record in records.values()} == {"santos_2015"}


def test_ram_usage_trace_exposes_both_selection_policies():
    trace = belief_assignment_trace(
        load_evidence_models(),
        "RAM Usage",
        MassAssignment.PUBLISHED_ANALOGUE,
    )
    assert trace["effect"] == "RAM Usage"
    assert trace["assignment"] == "published_analogue"
    assert len(trace["ordered_inputs"]) == RAM_USAGE_EVIDENCE_MODEL_COUNT
    assert trace["policies"]["evidence_factory_compat"]["result"]["intensity"] == ["SP"]
    assert trace["policies"]["santos_2015"]["result"]["intensity"] == ["PO", "SP"]


def test_ram_usage_trace_matches_committed_fixture():
    fixture_path = Path(__file__).parent / "fixtures" / "ram_usage_published_analogue_trace.json"
    expected = json.loads(fixture_path.read_text(encoding="utf-8"))
    actual = belief_assignment_trace(
        load_evidence_models(),
        "RAM Usage",
        MassAssignment.PUBLISHED_ANALOGUE,
    )
    assert actual == expected


def test_comparison_records_cover_published_table_effects():
    records = comparison_records()
    assert [record["effect"] for record in records] == [
        "Accuracy",
        "F1 Score",
        "mAP",
        "Storage Size",
        "GPU Utilization",
        "GPU Power Draw",
        "GPU Energy Consumption",
        "RAM Usage",
        "Inference Power Draw",
        "Inference Energy Consumption",
        "Inference Latency",
    ]
    accuracy = records[0]
    assert accuracy["n_primary_studies"] == ACCURACY_PRIMARY_STUDY_COUNT
    assert accuracy["n_evidence_models"] == ACCURACY_EVIDENCE_MODEL_COUNT
    assert {
        "mass_preserving_intensity",
        "mass_preserving_belief_percent",
        "mass_preserving_conflict",
        "equitable_intensity",
        "equitable_belief_percent",
        "equitable_conflict",
    } <= accuracy.keys()


def test_synthesis_row_reports_primary_study_count():
    models = load_evidence_models()
    row = synthesis_row(models, "Accuracy", MassAssignment.PUBLISHED_ANALOGUE)
    assert row.n_primary_studies == ACCURACY_PRIMARY_STUDY_COUNT
    assert row.n_evidence_models == ACCURACY_EVIDENCE_MODEL_COUNT
    assert contributing_study_ids(models, "Accuracy") == contributing_study_ids(models, "Accuracy")


def test_leave_one_study_out_gpu_energy_reports_all_omissions_when_belief_rises():
    records = leave_one_study_out_records()
    gpu_energy = [row for row in records if row.effect == "GPU Energy Consumption"]
    assert len(gpu_energy) == len({"S3", "S13", "S14"})
    assert all(row.belief_delta > 0 for row in gpu_energy)


def test_leave_one_study_out_worst_drop_for_accuracy_is_s13():
    records = {row.effect: row for row in leave_one_study_out_records()}
    accuracy = records["Accuracy"]
    assert accuracy.omitted_study_id == "S13"
    assert round(accuracy.full_belief, 2) == ACCURACY_LOO_FULL_BELIEF
    assert round(accuracy.loo_belief, 2) == ACCURACY_LOO_BELIEF
    assert accuracy.intensity_changed


def test_render_leave_one_study_out_table_includes_accuracy():
    latex = render_leave_one_study_out_table()
    assert "Accuracy" in latex
    assert "S13" in latex
    assert r"\begin{tabularx}" in latex


def test_write_leave_one_study_out_table_writes_fragment(tmp_path):
    path = write_leave_one_study_out_table(output_dir=tmp_path)
    assert path.name == "leave-one-study-out.tex"
    assert "Inf. latency" in path.read_text(encoding="utf-8")


def test_mass_preserving_sensitivity_rows_match_published_effects():
    rows = mass_preserving_sensitivity_rows()
    assert [row.effect for row in rows] == [expected.effect for expected in PUBLISHED_TABLE]


def test_render_sensitivity_mass_preserving_table_includes_accuracy():
    latex = render_sensitivity_mass_preserving_table()
    assert "Accuracy" in latex
    assert "Functional Suitability" in latex
    assert "83\\%" in latex or "83%" in latex


def test_write_sensitivity_mass_preserving_table_writes_fragment(tmp_path):
    path = write_sensitivity_mass_preserving_table(output_dir=tmp_path)
    assert path.name == "sensitivity-mass-preserving.tex"
    assert "Inf. latency" in path.read_text(encoding="utf-8")


def test_render_belief_assignment_table_orders_mass_preserving_before_equitable():
    latex = render_belief_assignment_table(
        [
            {
                "effect": "Accuracy",
                "n_primary_studies": 10,
                "n_evidence_models": 41,
                "published_intensity": "{WN, IF}",
                "published_belief_percent": 99,
                "published_conflict": 0.1894921993508596,
                "analogue_intensity": "{WN, IF}",
                "analogue_belief_percent": 99,
                "analogue_conflict": 0.1894921993508596,
                "unsplit_intensity": "IF",
                "unsplit_belief_percent": 99,
                "unsplit_conflict": 0.6670807973447248,
                "mass_preserving_intensity": "IF",
                "mass_preserving_belief_percent": 88,
                "mass_preserving_conflict": 0.12,
                "equitable_intensity": "IF",
                "equitable_belief_percent": 76,
                "equitable_conflict": 0.06027418047513863,
            },
            {
                "effect": "Storage Size",
                "n_primary_studies": 17,
                "n_evidence_models": 62,
                "published_intensity": "SP",
                "published_belief_percent": 100,
                "published_conflict": 6.13678329170885e-10,
                "analogue_intensity": "SP",
                "analogue_belief_percent": 100,
                "analogue_conflict": 6.13678329170885e-10,
                "unsplit_intensity": "SP",
                "unsplit_belief_percent": 100,
                "unsplit_conflict": 2.0653385096586093e-17,
                "mass_preserving_intensity": "SP",
                "mass_preserving_belief_percent": 100,
                "mass_preserving_conflict": 1e-6,
                "equitable_intensity": "SP",
                "equitable_belief_percent": 100,
                "equitable_conflict": 5.5137863460809415e-05,
            },
        ]
    )
    assert r"\begin{tabular}" in latex
    assert r"\{WN, IF\}" in latex
    assert "Accuracy" in latex
    assert "Storage size" in latex
    assert "0.99" in latex
    assert "0.19" in latex
    assert "0.67" in latex
    assert "0.88" in latex
    assert "0.12" in latex
    assert "0.76" in latex
    assert "0.06" in latex
    assert "1.00" in latex
    assert " & 1.00 & 0.00 & SP & 1.00 & 0.00 & SP & 1.00 & 0.00 & SP & 1.00 & 0.00" in latex
    assert r"\multicolumn{2}{c}{Contributors}" in latex
    assert "Accuracy & 10 & 41" in latex
    assert r"\multicolumn{3}{c}{Unsplit}" in latex
    assert r"\multicolumn{3}{c}{Mass-preserving}" in latex
    assert r"\multicolumn{3}{c}{Equitable}" in latex
    published_at = latex.index("Published analogue")
    unsplit_at = latex.index("Unsplit")
    mass_preserving_at = latex.index("Mass-preserving")
    equitable_at = latex.index("Equitable")
    assert published_at < unsplit_at < mass_preserving_at < equitable_at


def test_write_belief_assignment_table_writes_fragment(tmp_path):
    path = write_belief_assignment_table(
        [
            {
                "effect": "mAP",
                "n_primary_studies": 2,
                "n_evidence_models": 4,
                "published_intensity": "IF",
                "published_belief_percent": 45,
                "published_conflict": 0.31,
                "analogue_intensity": "IF",
                "analogue_belief_percent": 45,
                "analogue_conflict": 0.31,
                "unsplit_intensity": "IF",
                "unsplit_belief_percent": 61,
                "unsplit_conflict": 0.61,
                "mass_preserving_intensity": "IF",
                "mass_preserving_belief_percent": 50,
                "mass_preserving_conflict": 0.40,
                "equitable_intensity": "IF",
                "equitable_belief_percent": 38,
                "equitable_conflict": 0.21,
            }
        ],
        output_dir=tmp_path,
    )
    assert path.name == "belief-assignment.tex"
    text = path.read_text(encoding="utf-8")
    assert "mAP" in text
    assert "0.45" in text
    assert "0.31" in text
    assert "0.50" in text
    assert "0.40" in text
    assert "0.38" in text
