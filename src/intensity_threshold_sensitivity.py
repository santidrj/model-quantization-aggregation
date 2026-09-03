"""Intensity-threshold sensitivity tables for the published analogue (ADR 0014)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from src.belief_assignment import (
    EvidenceModel,
    MassAssignment,
    load_evidence_models,
    synthesis_row,
)
from src.config import PROCESSED_DATA_DIR, TABLES_DIR
from src.dempster_shafer import format_intensity
from src.discount_sensitivity import DiscountKey, load_discount_inputs
from src.effect_intensity import (
    CorrectnessMetrics,
    IntensityScale,
    PerformanceMetrics,
    ResourceEfficiencyMetrics,
    default_correctness_scale,
    default_resource_scale,
)

HEADLINE_EFFECTS: tuple[str, ...] = (
    "Accuracy",
    "Storage Size",
    "Inference Latency",
    "Inference Energy Consumption",
)

CORRECTNESS_METRICS: frozenset[str] = frozenset(CorrectnessMetrics.metrics())
RESOURCE_METRICS: frozenset[str] = frozenset(ResourceEfficiencyMetrics.metrics() + PerformanceMetrics.metrics())

ImprovementKey = DiscountKey


@dataclass(frozen=True)
class IntensityThresholdSpec:
    label: str
    is_reference: bool
    correctness_scale: IntensityScale
    resource_scale: IntensityScale
    remap_metrics: frozenset[str]


@dataclass(frozen=True)
class IntensityThresholdSensitivityRow:
    label: str
    spec: IntensityThresholdSpec
    is_reference: bool
    effect: str
    intensity: str
    belief_percent: int


def load_mean_relative_improvements(
    processed_root: Path | None = None,
) -> dict[ImprovementKey, float]:
    """Map (study, method, precision, effect) → mean relative improvement (%)."""
    return {key: mean for key, (_n_eff, _std, mean) in load_discount_inputs(processed_root).items()}


def remap_evidence_model_intensities(
    models: list[EvidenceModel],
    improvements: dict[ImprovementKey, float],
    *,
    scale: IntensityScale,
    metrics_on_scale: frozenset[str],
) -> list[EvidenceModel]:
    """Remap intensity labels from mean RIs; hold discounted support masses fixed."""
    remapped: list[EvidenceModel] = []
    for model in models:
        effects: dict[str, tuple[str, float]] = {}
        for effect, (intensity, belief) in model.effects.items():
            if effect not in metrics_on_scale:
                effects[effect] = (intensity, belief)
                continue
            key = (model.study_id, model.quantization_method, model.precision_configuration, effect)
            if key not in improvements:
                effects[effect] = (intensity, belief)
                continue
            effects[effect] = (scale.get_intensity(improvements[key]), belief)
        remapped.append(replace(model, effects=effects))
    return remapped


def consensus_threshold_specs() -> list[IntensityThresholdSpec]:
    """Published cuts plus consensus literature alternates (one cut at a time)."""
    published_correctness = default_correctness_scale()
    published_resource = default_resource_scale()
    return [
        IntensityThresholdSpec(
            label="published cuts",
            is_reference=True,
            correctness_scale=published_correctness,
            resource_scale=published_resource,
            remap_metrics=frozenset(),
        ),
        IntensityThresholdSpec(
            label=r"functional-suitability indifferent $=1$",
            is_reference=False,
            correctness_scale=default_correctness_scale(weak_indifferent_effect=1),
            resource_scale=published_resource,
            remap_metrics=CORRECTNESS_METRICS,
        ),
        IntensityThresholdSpec(
            label=r"resource/performance strong $=75$",
            is_reference=False,
            correctness_scale=published_correctness,
            resource_scale=default_resource_scale(strong_effect=75),
            remap_metrics=RESOURCE_METRICS,
        ),
    ]


def _models_for_spec(
    models: list[EvidenceModel],
    improvements: dict[ImprovementKey, float],
    spec: IntensityThresholdSpec,
) -> list[EvidenceModel]:
    if not spec.remap_metrics:
        return models
    if spec.remap_metrics <= CORRECTNESS_METRICS:
        scale = spec.correctness_scale
    elif spec.remap_metrics <= RESOURCE_METRICS:
        scale = spec.resource_scale
    else:
        raise ValueError(f"Mixed remap metrics are not supported: {sorted(spec.remap_metrics)}")
    return remap_evidence_model_intensities(
        models,
        improvements,
        scale=scale,
        metrics_on_scale=spec.remap_metrics,
    )


def intensity_threshold_sensitivity_rows(
    models: list[EvidenceModel],
    improvements: dict[ImprovementKey, float],
    *,
    effects: tuple[str, ...] = HEADLINE_EFFECTS,
    specs: list[IntensityThresholdSpec] | None = None,
) -> list[IntensityThresholdSensitivityRow]:
    rows: list[IntensityThresholdSensitivityRow] = []
    for spec in specs if specs is not None else consensus_threshold_specs():
        remapped = _models_for_spec(models, improvements, spec)
        for effect in effects:
            synthesized = synthesis_row(remapped, effect, MassAssignment.PUBLISHED_ANALOGUE)
            rows.append(
                IntensityThresholdSensitivityRow(
                    label=spec.label,
                    spec=spec,
                    is_reference=spec.is_reference,
                    effect=effect,
                    intensity=format_intensity(synthesized.intensity),
                    belief_percent=synthesized.belief_percent,
                )
            )
    return rows


def render_intensity_threshold_sensitivity_table(
    rows: list[IntensityThresholdSensitivityRow],
) -> str:
    lines = [
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Setting & Effect & Aggregated intensity & Belief \\",
        r"\midrule",
    ]
    for row in rows:
        marker = r"\textbf{reference} " if row.is_reference else ""
        lines.append(f"{marker}{row.label} & {row.effect} & {row.intensity} & {row.belief_percent}\\% \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_intensity_threshold_sensitivity_tables(
    *,
    processed_root: Path | None = None,
    output_dir: Path | None = None,
    models: list[EvidenceModel] | None = None,
) -> list[Path]:
    directory = output_dir or TABLES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    loaded_models = models if models is not None else load_evidence_models(processed_root)
    improvements = load_mean_relative_improvements(processed_root or PROCESSED_DATA_DIR)
    rows = intensity_threshold_sensitivity_rows(loaded_models, improvements)
    path = directory / "intensity-threshold-sensitivity.tex"
    path.write_text(render_intensity_threshold_sensitivity_table(rows), encoding="utf-8")
    return [path]
