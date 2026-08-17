"""Equitable belief split: study-belief assignment across evidence models."""

from src.equitable_belief_split import (
    EvidenceModel,
    MassAssignment,
    assigned_mass,
    comparison_records,
    load_evidence_models,
    pieces_for_effect,
    reproduction_mismatches,
)

AJI_STUDY_BELIEF = 0.6720808333333333
AJI_EVIDENCE_MODEL_COUNT = 4
AJI_PROCESSED_BELIEF = 0.327
CORPUS_EVIDENCE_MODEL_COUNT = 76
ACCURACY_EVIDENCE_MODEL_COUNT = 41


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


def test_published_analogue_matches_intensity_and_model_counts():
    assert reproduction_mismatches(checks=("intensity", "n_evidence_models")) == []


def test_published_analogue_belief_matches_except_inference_power_draw():
    mismatches = reproduction_mismatches(checks=("belief",))
    assert mismatches == ["Inference Power Draw: belief 71% != 72%"]


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
    assert accuracy["n_evidence_models"] == ACCURACY_EVIDENCE_MODEL_COUNT
