"""Intensity-threshold sensitivity: remap RI→intensity; hold discounted support mass fixed."""

import json
from pathlib import Path

from src.belief_assignment import EvidenceModel, load_evidence_models
from src.effect_intensity import (
    CorrectnessIntensity,
    EffectIntensity,
    IntensityScale,
    default_correctness_scale,
    default_resource_scale,
)
from src.intensity_threshold_sensitivity import (
    IntensityThresholdSensitivityRow,
    IntensityThresholdSpec,
    intensity_threshold_sensitivity_rows,
    load_mean_relative_improvements,
    remap_evidence_model_intensities,
    render_intensity_threshold_sensitivity_table,
)


def test_correctness_scale_indifferent_one_moves_boundary_crossing_ri():
    """|RI|=1.5 is IF at published indifferent=2 and WN-IF when indifferent=1."""
    published = default_correctness_scale()
    tight = default_correctness_scale(weak_indifferent_effect=1)
    assert published.get_intensity(-1.5) == "indifferent"
    assert tight.get_intensity(-1.5) == "weakly negative - indifferent"


def test_remap_changes_correctness_intensity_and_holds_discounted_support_mass():
    model = EvidenceModel(
        study_id="S14",
        quantization_method="ptq",
        precision_configuration="w-int4",
        study_belief=0.713,
        evidence_model_count=1,
        effects={
            "Accuracy": ("indifferent", 0.031),
            "Storage Size": ("strongly positive", 0.712),
        },
    )
    improvements = {
        ("S14", "ptq", "w-int4", "Accuracy"): -1.5,
        ("S14", "ptq", "w-int4", "Storage Size"): 75.0,
    }
    remapped = remap_evidence_model_intensities(
        [model],
        improvements,
        scale=default_correctness_scale(weak_indifferent_effect=1),
        metrics_on_scale=frozenset({"Accuracy"}),
    )
    assert remapped[0].effects["Accuracy"] == ("weakly negative - indifferent", 0.031)
    assert remapped[0].effects["Storage Size"] == ("strongly positive", 0.712)


def test_resource_strong_75_moves_mid_band_off_strongly_positive():
    scale = default_resource_scale(strong_effect=75)
    assert default_resource_scale().get_intensity(60.0) == "strongly positive"
    assert scale.get_intensity(60.0) == "positive - strongly positive"
    assert scale.get_intensity(75.0) == "positive - strongly positive"
    assert scale.get_intensity(76.0) == "strongly positive"


def test_sensitivity_rows_include_reference_and_consensus_alts():
    model = EvidenceModel(
        study_id="S1",
        quantization_method="ptq",
        precision_configuration="w-int8",
        study_belief=0.5,
        evidence_model_count=1,
        effects={
            "Accuracy": ("indifferent", 0.5),
            "Storage Size": ("strongly positive", 0.5),
            "Inference Latency": ("positive - strongly positive", 0.5),
            "Inference Energy Consumption": ("strongly positive", 0.5),
        },
    )
    improvements = {
        ("S1", "ptq", "w-int8", "Accuracy"): 0.0,
        ("S1", "ptq", "w-int8", "Storage Size"): 80.0,
        ("S1", "ptq", "w-int8", "Inference Latency"): 60.0,
        ("S1", "ptq", "w-int8", "Inference Energy Consumption"): 80.0,
    }
    rows = intensity_threshold_sensitivity_rows([model], improvements)
    settings = [row.label for row in rows if row.effect == "Accuracy"]
    assert settings == [
        "published cuts",
        r"functional-suitability indifferent $=1$",
        r"resource/performance strong $=75$",
    ]
    assert [row.is_reference for row in rows if row.effect == "Accuracy"] == [True, False, False]
    assert {row.effect for row in rows} == {
        "Accuracy",
        "Storage Size",
        "Inference Latency",
        "Inference Energy Consumption",
    }


def test_render_marks_reference_and_lists_effect_columns():
    row = IntensityThresholdSensitivityRow(
        label="published cuts",
        spec=IntensityThresholdSpec(
            label="published cuts",
            is_reference=True,
            correctness_scale=default_correctness_scale(),
            resource_scale=default_resource_scale(),
            remap_metrics=frozenset(),
        ),
        is_reference=True,
        effect="Accuracy",
        intensity="{IF}",
        belief_percent=99,
    )
    latex = render_intensity_threshold_sensitivity_table([row])
    assert r"\textbf{reference}" in latex
    assert "Accuracy" in latex
    assert "99\\%" in latex


def test_intensity_scale_matches_singleton_defaults():
    ri = 12.0
    assert default_correctness_scale().get_intensity(ri) == CorrectnessIntensity().get_intensity(ri)
    assert default_resource_scale().get_intensity(ri) == EffectIntensity().get_intensity(ri)
    assert isinstance(default_correctness_scale(), IntensityScale)


def test_reference_rows_match_validated_synthesis_headline_effects():
    validated = json.loads(Path("data/processed/validated-synthesis.json").read_text(encoding="utf-8"))
    rows = intensity_threshold_sensitivity_rows(
        load_evidence_models(),
        load_mean_relative_improvements(),
    )
    reference = {row.effect: row for row in rows if row.is_reference}
    for effect in (
        "Accuracy",
        "Storage Size",
        "Inference Latency",
        "Inference Energy Consumption",
    ):
        expected = validated["effects"][effect]
        assert reference[effect].intensity == expected["intensity_label"]
        assert reference[effect].belief_percent == expected["belief_percent"]
