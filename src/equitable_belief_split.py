"""Equitable belief split: study-belief masses on evidence models and Dempster–Shafer synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from src.config import processed_paper_path
from src.data.papers.entities import Papers
from src.data.papers.study_id import study_id_sort_key
from src.dempster_shafer import combine_effect, format_intensity


class MassAssignment(Enum):
    PUBLISHED_ANALOGUE = "published_analogue"
    UNDISCOUNTED_UNSPLIT = "undiscounted_unsplit"
    EQUITABLE_BELIEF_SPLIT = "equitable_belief_split"


@dataclass(frozen=True)
class EvidenceModel:
    study_id: str
    quantization_method: str
    precision_configuration: str
    study_belief: float
    evidence_model_count: int
    effects: dict[str, tuple[str, float]]


def assigned_mass(model: EvidenceModel, assignment: MassAssignment, effect: str | None = None) -> float:
    if assignment is MassAssignment.EQUITABLE_BELIEF_SPLIT:
        return model.study_belief / model.evidence_model_count
    if assignment is MassAssignment.UNDISCOUNTED_UNSPLIT:
        return model.study_belief
    if assignment is MassAssignment.PUBLISHED_ANALOGUE:
        if effect is None:
            raise ValueError("Published analogue mass requires an effect")
        return model.effects[effect][1]
    raise ValueError(f"Unknown mass assignment: {assignment}")


def _model_sort_key(model: EvidenceModel) -> tuple:
    return (*study_id_sort_key(model.study_id), model.quantization_method, model.precision_configuration)


def pieces_for_effect(
    models: list[EvidenceModel],
    effect: str,
    assignment: MassAssignment,
) -> list[tuple[str, float]]:
    pieces: list[tuple[str, float]] = []
    for model in sorted(models, key=_model_sort_key):
        if effect not in model.effects:
            continue
        intensity, _processed_belief = model.effects[effect]
        pieces.append((intensity, assigned_mass(model, assignment, effect=effect)))
    return pieces


_MODEL_IDENTITY_KEYS = frozenset({"quantization_method", "precision_configuration"})


def _effect_name(metric_key: str) -> str:
    titled = metric_key.replace("_", " ").title()
    return (
        titled.replace("Gpu", "GPU")
        .replace("Ram", "RAM")
        .replace("Dsc", "DSC")
        .replace("Miou", "mIoU")
        .replace("Map 5 95", "mAP@0.5:0.95")
        .replace("Map 5", "mAP@0.5")
        .replace("Map", "mAP")
        .replace("Bleu", "BLEU")
    )


def _effects_from_record(record: dict) -> dict[str, tuple[str, float]]:
    effects: dict[str, tuple[str, float]] = {}
    for key, payload in record.items():
        if key in _MODEL_IDENTITY_KEYS or not isinstance(payload, dict):
            continue
        intensity = payload.get("intensity")
        belief = payload.get("belief")
        if intensity is None or belief is None:
            continue
        effects[_effect_name(key)] = (str(intensity), float(belief))
    return effects


def load_evidence_models(processed_root: Path | None = None) -> list[EvidenceModel]:
    """Load one evidence model per by-precision record in processed JSON."""
    models: list[EvidenceModel] = []
    for paper in Papers:
        path = (
            processed_root / paper.value.KEY / "effects_by_precision.json"
            if processed_root is not None
            else processed_paper_path(paper.value.KEY, "effects_by_precision.json")
        )
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        evidence_model_count = len(records)
        for record in records:
            models.append(
                EvidenceModel(
                    study_id=paper.value.ID,
                    quantization_method=record["quantization_method"],
                    precision_configuration=record["precision_configuration"],
                    study_belief=paper.value.BELIEF,
                    evidence_model_count=evidence_model_count,
                    effects=_effects_from_record(record),
                )
            )
    return models


@dataclass(frozen=True)
class PublishedRow:
    effect: str
    intensity: frozenset[str]
    belief_percent: int
    conflict: float
    n_evidence_models: int
    conflict_abs: float


PUBLISHED_TABLE: tuple[PublishedRow, ...] = (
    PublishedRow("Accuracy", frozenset({"WN", "IF"}), 99, 0.14, 41, 0.005),
    PublishedRow("F1 Score", frozenset({"IF"}), 75, 0.15, 9, 0.005),
    PublishedRow("mAP", frozenset({"IF"}), 45, 0.31, 4, 0.005),
    PublishedRow("Storage Size", frozenset({"SP"}), 100, 1.09e-8, 62, 5e-10),
    PublishedRow("GPU Utilization", frozenset({"IF"}), 74, 0.0, 3, 1e-12),
    PublishedRow("GPU Power Draw", frozenset({"IF", "WP"}), 98, 0.30, 5, 0.005),
    PublishedRow("GPU Energy Consumption", frozenset({"SP"}), 74, 0.12, 5, 0.005),
    PublishedRow("RAM Usage", frozenset({"SP"}), 47, 0.20, 3, 0.005),
    PublishedRow("Inference Power Draw", frozenset({"WP"}), 72, 0.06, 10, 0.005),
    PublishedRow("Inference Energy Consumption", frozenset({"SP"}), 100, 4e-3, 27, 5e-4),
    PublishedRow("Inference Latency", frozenset({"PO", "SP"}), 100, 0.44, 51, 0.005),
)


@dataclass(frozen=True)
class SynthesisRow:
    effect: str
    intensity: frozenset[str]
    belief: float
    belief_percent: int
    conflict: float
    n_evidence_models: int


def synthesis_row(
    models: list[EvidenceModel],
    effect: str,
    assignment: MassAssignment,
) -> SynthesisRow:
    pieces = pieces_for_effect(models, effect, assignment)
    if not pieces:
        raise ValueError(f"No evidence models report {effect}")
    combined = combine_effect(pieces)
    return SynthesisRow(
        effect=effect,
        intensity=combined.intensity,
        belief=combined.belief,
        belief_percent=round(combined.belief * 100),
        conflict=combined.conflict,
        n_evidence_models=len(pieces),
    )


def reproduction_mismatches(
    models: list[EvidenceModel] | None = None,
    *,
    checks: tuple[str, ...] = ("intensity", "belief", "conflict", "n_evidence_models"),
) -> list[str]:
    """Return human-readable gate failures for the published analogue vs Table general-results."""
    loaded = models if models is not None else load_evidence_models()
    mismatches: list[str] = []
    for expected in PUBLISHED_TABLE:
        actual = synthesis_row(loaded, expected.effect, MassAssignment.PUBLISHED_ANALOGUE)
        if "n_evidence_models" in checks and actual.n_evidence_models != expected.n_evidence_models:
            mismatches.append(
                f"{expected.effect}: n_evidence_models {actual.n_evidence_models} != {expected.n_evidence_models}"
            )
        if "intensity" in checks and actual.intensity != expected.intensity:
            mismatches.append(
                f"{expected.effect}: intensity "
                f"{format_intensity(actual.intensity)} != {format_intensity(expected.intensity)}"
            )
        if "belief" in checks and actual.belief_percent != expected.belief_percent:
            mismatches.append(f"{expected.effect}: belief {actual.belief_percent}% != {expected.belief_percent}%")
        if "conflict" in checks and abs(actual.conflict - expected.conflict) > expected.conflict_abs:
            mismatches.append(
                f"{expected.effect}: conflict {actual.conflict} != {expected.conflict} (±{expected.conflict_abs})"
            )
    return mismatches


def comparison_records(models: list[EvidenceModel] | None = None) -> list[dict[str, object]]:
    """One row per published-table effect with analogue, unsplit, and split synthesis."""
    loaded = models if models is not None else load_evidence_models()
    records: list[dict[str, object]] = []
    for expected in PUBLISHED_TABLE:
        analogue = synthesis_row(loaded, expected.effect, MassAssignment.PUBLISHED_ANALOGUE)
        unsplit = synthesis_row(loaded, expected.effect, MassAssignment.UNDISCOUNTED_UNSPLIT)
        split = synthesis_row(loaded, expected.effect, MassAssignment.EQUITABLE_BELIEF_SPLIT)
        records.append(
            {
                "effect": expected.effect,
                "n_evidence_models": analogue.n_evidence_models,
                "analogue_intensity": format_intensity(analogue.intensity),
                "analogue_belief_percent": analogue.belief_percent,
                "analogue_conflict": analogue.conflict,
                "unsplit_intensity": format_intensity(unsplit.intensity),
                "unsplit_belief_percent": unsplit.belief_percent,
                "unsplit_conflict": unsplit.conflict,
                "split_intensity": format_intensity(split.intensity),
                "split_belief_percent": split.belief_percent,
                "split_conflict": split.conflict,
                "published_intensity": format_intensity(expected.intensity),
                "published_belief_percent": expected.belief_percent,
                "published_conflict": expected.conflict,
            }
        )
    return records
