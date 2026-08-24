"""Write-back of effect intensity, discount residual, and supporting statistics to Evidence Factory evidence editors."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any
from urllib.request import urlopen

from src.belief_assignment import (  # noqa: PLC2701 — same join as evidence-model loading
    _MODEL_IDENTITY_KEYS,
    _effect_name,
    _evidence_factory_ids_by_study,
)
from src.config import PROCESSED_DATA_DIR
from src.data.papers.entities import Papers

_SUPPORTING_STATISTICS_KEYS = (
    "improvement",
    "std",
    "iqr",
    "sample_size_discount",
    "variability_discount",
    "discount_factor",
    "p_value",
)

_CAUSE_EFFECT_INTENSITY_BY_PROPOSITION: dict[tuple[int, float], str] = {
    (64, -3.0): "Strongly Negative",
    (26, -2.5): "Strongly Negative - Negative",
    (106, -2.0): "Negative",
    (88, -1.5): "Negative - Weakly Negative",
    (120, -1.0): "Weakly Negative",
    (136, -0.5): "Weakly Negative - Indifferent",
    (30, 0.0): "Indifferent",
    (138, 0.5): "Indifferent - Weakly Positive",
    (122, 1.0): "Weakly Positive",
    (121, 1.5): "Weakly Positive - Positive",
    (139, 2.0): "Positive",
    (65, 2.5): "Positive - Strongly Positive",
    (66, 3.0): "Strongly Positive",
}

EVIDENCE_DATA_URL = "https://evidencefactory.lens-ese.cos.ufrj.br/evidenceEditor/evidencedata"


def intensity_control_option(processed_intensity: str) -> str:
    """Title-case phrase matching the evidence editor intensity drop-down."""
    return processed_intensity.title()


def supporting_statistics_comment(payload: dict[str, Any]) -> str:
    """JSON effect comment holding supporting statistics, not belief or intensity."""
    return json.dumps({key: payload.get(key) for key in _SUPPORTING_STATISTICS_KEYS}, indent=2)


@dataclass(frozen=True)
class DesiredEffect:
    label: str
    complete: bool
    intensity_option: str | None
    p_value: float | None
    comment: str | None


@dataclass(frozen=True)
class MappingIntegrityFault:
    study_id: str
    kind: str
    evidence_factory_id: int | None = None


def mapping_integrity_faults(
    *,
    mapping_ids_by_study: dict[str, list[int]],
    record_counts_by_study: dict[str, int],
) -> tuple[MappingIntegrityFault, ...]:
    """Faults that refuse apply for the whole corpus: duplicate IDs or length mismatch."""
    occurrences: dict[int, list[str]] = {}
    for study_id, factory_ids in mapping_ids_by_study.items():
        for factory_id in factory_ids:
            occurrences.setdefault(factory_id, []).append(study_id)

    faults: list[MappingIntegrityFault] = []
    for study_id, factory_ids in mapping_ids_by_study.items():
        record_count = record_counts_by_study.get(study_id, 0)
        if len(factory_ids) != record_count:
            faults.append(MappingIntegrityFault(study_id=study_id, kind="length_mismatch"))
        seen_in_study: set[int] = set()
        for factory_id in factory_ids:
            if factory_id in seen_in_study or len(occurrences[factory_id]) > 1:
                faults.append(
                    MappingIntegrityFault(
                        study_id=study_id,
                        kind="duplicate_id",
                        evidence_factory_id=factory_id,
                    )
                )
            seen_in_study.add(factory_id)
    return tuple(faults)


def write_back_is_allowed(faults: tuple[MappingIntegrityFault, ...]) -> bool:
    return not faults


@dataclass(frozen=True)
class DesiredModel:
    study_id: str
    evidence_factory_id: int | None
    quantization_method: str
    precision_configuration: str
    effects: tuple[DesiredEffect, ...]


@dataclass(frozen=True)
class LiveElement:
    label: str
    kind: str
    intensity: str | None
    p_value: float | None
    comment: str | None


@dataclass(frozen=True)
class EffectDelta:
    label: str
    intensity_option: str
    p_value: float
    comment: str


@dataclass(frozen=True)
class ModelPlan:
    study_id: str
    evidence_factory_id: int | None
    deltas: tuple[EffectDelta, ...]
    unmatched_local_effects: tuple[str, ...]
    extra_effect_nodes: tuple[str, ...]
    incomplete_effects: tuple[str, ...]
    ambiguous_local_effects: tuple[str, ...] = ()


def _comments_match(desired: str, live: str | None) -> bool:
    if live is None:
        return False
    try:
        return json.loads(desired) == json.loads(live)
    except json.JSONDecodeError:
        return desired == live


def plan_model(desired: DesiredModel, live_elements: tuple[LiveElement, ...]) -> ModelPlan:
    """Diff one evidence model against live evidence-editor elements."""
    effect_nodes = [element for element in live_elements if element.kind.lower() == "effect"]
    nodes_by_label: dict[str, list[LiveElement]] = {}
    for node in effect_nodes:
        nodes_by_label.setdefault(node.label, []).append(node)

    desired_labels = {effect.label for effect in desired.effects}
    extra = tuple(node.label for node in effect_nodes if node.label not in desired_labels)

    deltas: list[EffectDelta] = []
    unmatched: list[str] = []
    ambiguous: list[str] = []
    incomplete = tuple(effect.label for effect in desired.effects if not effect.complete)

    for effect in desired.effects:
        if not effect.complete:
            continue
        matches = nodes_by_label.get(effect.label, [])
        if len(matches) == 0:
            unmatched.append(effect.label)
            continue
        if len(matches) > 1:
            ambiguous.append(effect.label)
            continue
        live = matches[0]
        comment = effect.comment or ""
        if (
            live.intensity != effect.intensity_option
            or live.p_value != effect.p_value
            or not _comments_match(comment, live.comment)
        ):
            deltas.append(
                EffectDelta(
                    label=effect.label,
                    intensity_option=effect.intensity_option or "",
                    p_value=effect.p_value or 0.0,
                    comment=comment,
                )
            )

    return ModelPlan(
        study_id=desired.study_id,
        evidence_factory_id=desired.evidence_factory_id,
        deltas=tuple(deltas),
        unmatched_local_effects=tuple(unmatched),
        extra_effect_nodes=extra,
        incomplete_effects=incomplete,
        ambiguous_local_effects=tuple(ambiguous),
    )


def desired_effect(metric_key: str, payload: dict[str, Any]) -> DesiredEffect:
    """One evidence-model effect as it should appear in the evidence editor."""
    label = _effect_name(metric_key)
    intensity = payload.get("intensity")
    improvement = payload.get("improvement")
    if intensity is None or improvement is None:
        return DesiredEffect(label=label, complete=False, intensity_option=None, p_value=None, comment=None)
    return DesiredEffect(
        label=label,
        complete=True,
        intensity_option=intensity_control_option(str(intensity)),
        p_value=float(payload["p_value"]),
        comment=supporting_statistics_comment(payload),
    )


def live_elements_from_evidence_dto(dto: dict[str, Any]) -> tuple[LiveElement, ...]:
    """Map Evidence Factory evidence JSON to live Effect nodes (CAUSES targets)."""
    elements: list[LiveElement] = []
    for relationship in dto.get("relationships", []):
        if relationship.get("type") != "CAUSES":
            continue
        to_term = relationship.get("toTerm") or {}
        label = to_term.get("name")
        if not label:
            continue
        proposition_id = relationship.get("propositionId")
        proposition_order = relationship.get("propositionOrder")
        intensity = None
        if proposition_id is not None and proposition_order is not None:
            intensity = _CAUSE_EFFECT_INTENSITY_BY_PROPOSITION.get((int(proposition_id), float(proposition_order)))
        p_value = relationship.get("pValue")
        elements.append(
            LiveElement(
                label=str(label),
                kind="Effect",
                intensity=intensity,
                p_value=None if p_value is None else float(p_value),
                comment=relationship.get("explanation"),
            )
        )
    return tuple(elements)


def _proposition_for_intensity(intensity_option: str) -> tuple[int, float]:
    for proposition, label in _CAUSE_EFFECT_INTENSITY_BY_PROPOSITION.items():
        if label == intensity_option:
            return proposition
    raise ValueError(f"No Evidence Factory intensity option named {intensity_option!r}")


def apply_deltas_to_evidence_dto(dto: dict[str, Any], deltas: tuple[EffectDelta, ...]) -> dict[str, Any]:
    """Patch CAUSES rows in an evidence DTO; leave primary belief unchanged."""
    patched = deepcopy(dto)
    label_counts: dict[str, int] = {}
    for relationship in patched.get("relationships", []):
        if relationship.get("type") != "CAUSES":
            continue
        name = (relationship.get("toTerm") or {}).get("name")
        if name:
            label_counts[name] = label_counts.get(name, 0) + 1
    by_label = {delta.label: delta for delta in deltas}
    for relationship in patched.get("relationships", []):
        if relationship.get("type") != "CAUSES":
            continue
        name = (relationship.get("toTerm") or {}).get("name")
        if label_counts.get(name, 0) != 1:
            continue
        delta = by_label.get(name)
        if delta is None:
            continue
        proposition_id, proposition_order = _proposition_for_intensity(delta.intensity_option)
        relationship["propositionId"] = proposition_id
        relationship["propositionOrder"] = proposition_order
        relationship["pValue"] = delta.p_value
        relationship["explanation"] = delta.comment
    return patched


@dataclass(frozen=True)
class WriteBackCatalog:
    models: tuple[DesiredModel, ...]
    mapping_ids_by_study: dict[str, list[int]]
    record_counts_by_study: dict[str, int]

    @property
    def faults(self) -> tuple[MappingIntegrityFault, ...]:
        return mapping_integrity_faults(
            mapping_ids_by_study=self.mapping_ids_by_study,
            record_counts_by_study=self.record_counts_by_study,
        )


def _duplicate_factory_ids(mapping_ids_by_study: dict[str, list[int]]) -> set[int]:
    counts: dict[int, int] = {}
    for factory_ids in mapping_ids_by_study.values():
        for factory_id in factory_ids:
            counts[factory_id] = counts.get(factory_id, 0) + 1
    return {factory_id for factory_id, count in counts.items() if count > 1}


def live_read_models(catalog: WriteBackCatalog) -> tuple[DesiredModel, ...]:
    """Models whose study mapping is intact and whose editor ID is unique."""
    blocked_studies = {fault.study_id for fault in catalog.faults}
    blocked_ids = _duplicate_factory_ids(catalog.mapping_ids_by_study)
    return tuple(
        model
        for model in catalog.models
        if model.study_id not in blocked_studies
        and model.evidence_factory_id is not None
        and model.evidence_factory_id not in blocked_ids
    )


def plan_catalog(
    catalog: WriteBackCatalog,
    live_by_id: dict[int, tuple[LiveElement, ...]],
) -> tuple[ModelPlan, ...]:
    plans: list[ModelPlan] = []
    for model in live_read_models(catalog):
        factory_id = model.evidence_factory_id
        if factory_id not in live_by_id:
            continue
        plans.append(plan_model(model, live_by_id[factory_id]))
    return tuple(plans)


@dataclass(frozen=True)
class ApplyReport:
    refused: bool
    faults: tuple[MappingIntegrityFault, ...]
    written: tuple[tuple[int, str], ...]
    stopped_on_error: str | None
    plans: tuple[ModelPlan, ...]


class EvidenceEditorPort:
    def read_effects(self, evidence_factory_id: int) -> tuple[LiveElement, ...]:
        raise NotImplementedError

    def write_effect(self, evidence_factory_id: int, delta: EffectDelta) -> None:
        raise NotImplementedError


def apply_write_back(catalog: WriteBackCatalog, editor: EvidenceEditorPort) -> ApplyReport:
    """Write deltas in catalog order. Refused while mapping integrity fails. Stops on first write error."""
    faults = catalog.faults
    if not write_back_is_allowed(faults):
        return ApplyReport(refused=True, faults=faults, written=(), stopped_on_error=None, plans=())

    written: list[tuple[int, str]] = []
    plans: list[ModelPlan] = []
    for model in live_read_models(catalog):
        factory_id = model.evidence_factory_id
        if factory_id is None:
            continue
        plan = plan_model(model, editor.read_effects(factory_id))
        plans.append(plan)
        for delta in plan.deltas:
            try:
                editor.write_effect(factory_id, delta)
            except Exception as error:  # noqa: BLE001 — editor adapter errors must halt apply
                return ApplyReport(
                    refused=False,
                    faults=faults,
                    written=tuple(written),
                    stopped_on_error=str(error),
                    plans=tuple(plans),
                )
            written.append((factory_id, delta.label))
    return ApplyReport(refused=False, faults=faults, written=tuple(written), stopped_on_error=None, plans=tuple(plans))


def models_from_precision_records(
    study_id: str,
    records: list[dict[str, Any]],
    factory_ids: list[int],
) -> tuple[DesiredModel, ...]:
    models: list[DesiredModel] = []
    for index, record in enumerate(records):
        factory_id = factory_ids[index] if index < len(factory_ids) else None
        effects = tuple(
            desired_effect(key, payload)
            for key, payload in record.items()
            if key not in _MODEL_IDENTITY_KEYS and isinstance(payload, dict)
        )
        models.append(
            DesiredModel(
                study_id=study_id,
                evidence_factory_id=factory_id,
                quantization_method=str(record["quantization_method"]),
                precision_configuration=str(record["precision_configuration"]),
                effects=effects,
            )
        )
    return tuple(models)


def load_write_back_catalog(processed_root: Path | None = None) -> WriteBackCatalog:
    """Join by-precision effects to evidence-editor IDs by mapping list index."""
    root = processed_root if processed_root is not None else PROCESSED_DATA_DIR
    mapping_ids_by_study = _evidence_factory_ids_by_study(root / "evidence-diagrams-mapping.md")
    models: list[DesiredModel] = []
    record_counts_by_study: dict[str, int] = {}
    for paper in Papers:
        records = json.loads((root / paper.value.KEY / "effects_by_precision.json").read_text(encoding="utf-8"))
        study_id = paper.value.ID
        record_counts_by_study[study_id] = len(records)
        models.extend(models_from_precision_records(study_id, records, mapping_ids_by_study.get(study_id, [])))
    return WriteBackCatalog(
        models=tuple(models),
        mapping_ids_by_study=mapping_ids_by_study,
        record_counts_by_study=record_counts_by_study,
    )


def fetch_live_elements(evidence_factory_id: int) -> tuple[LiveElement, ...]:
    url = f"{EVIDENCE_DATA_URL}?evidenceId={evidence_factory_id}"
    with urlopen(url, timeout=60) as response:  # noqa: S310 — Evidence Factory public evidence JSON
        payload = json.loads(response.read().decode("utf-8"))
    return live_elements_from_evidence_dto(payload)


def catalog_as_dict(catalog: WriteBackCatalog) -> dict[str, Any]:
    live_ids = {model.evidence_factory_id for model in live_read_models(catalog)}
    return {
        "apply_allowed": write_back_is_allowed(catalog.faults),
        "faults": [asdict(fault) for fault in catalog.faults],
        "models": [
            {
                "study_id": model.study_id,
                "evidence_factory_id": model.evidence_factory_id,
                "quantization_method": model.quantization_method,
                "precision_configuration": model.precision_configuration,
                "live_read": model.evidence_factory_id in live_ids,
                "effects": [asdict(effect) for effect in model.effects],
            }
            for model in catalog.models
        ],
    }


def plan_as_dict(plan: ModelPlan) -> dict[str, Any]:
    return {
        "study_id": plan.study_id,
        "evidence_factory_id": plan.evidence_factory_id,
        "deltas": [asdict(delta) for delta in plan.deltas],
        "unmatched_local_effects": list(plan.unmatched_local_effects),
        "extra_effect_nodes": list(plan.extra_effect_nodes),
        "incomplete_effects": list(plan.incomplete_effects),
        "ambiguous_local_effects": list(plan.ambiguous_local_effects),
    }


def collect_live_plans(catalog: WriteBackCatalog) -> tuple[ModelPlan, ...]:
    live_by_id: dict[int, tuple[LiveElement, ...]] = {}
    for model in live_read_models(catalog):
        factory_id = model.evidence_factory_id
        if factory_id is None:
            continue
        live_by_id[factory_id] = fetch_live_elements(factory_id)
    return plan_catalog(catalog, live_by_id)


def render_plan(catalog: WriteBackCatalog, plans: tuple[ModelPlan, ...]) -> str:
    lines: list[str] = []
    if catalog.faults:
        lines.append("Mapping integrity faults (apply refused for the whole corpus):")
        for fault in catalog.faults:
            extra = f" id={fault.evidence_factory_id}" if fault.evidence_factory_id is not None else ""
            lines.append(f"  {fault.study_id} {fault.kind}{extra}")
        lines.append("")
    if not plans:
        lines.append("No intact studies to live-read.")
        return "\n".join(lines) + "\n"
    for plan in plans:
        lines.append(f"{plan.study_id} editor {plan.evidence_factory_id}")
        if plan.incomplete_effects:
            lines.append(f"  incomplete (skipped): {', '.join(plan.incomplete_effects)}")
        if plan.unmatched_local_effects:
            lines.append(f"  unmatched local effects: {', '.join(plan.unmatched_local_effects)}")
        if plan.ambiguous_local_effects:
            lines.append(f"  ambiguous effect labels: {', '.join(plan.ambiguous_local_effects)}")
        if plan.extra_effect_nodes:
            lines.append(f"  extra effect nodes: {', '.join(plan.extra_effect_nodes)}")
        if plan.deltas:
            for delta in plan.deltas:
                lines.append(
                    f"  delta {delta.label}: intensity={delta.intensity_option} "
                    f"discount_residual={delta.p_value} comment=replaced"
                )
        else:
            lines.append("  no deltas")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in {"catalog", "plan", "patch-dto"}:
        sys.stderr.write("usage: python -m src.evidence_editor_sync {catalog|plan|patch-dto}\n")
        return 2
    command = args[0]
    if command == "patch-dto":
        payload = json.loads(sys.stdin.read())
        deltas = tuple(
            EffectDelta(
                label=item["label"],
                intensity_option=item["intensity_option"],
                p_value=float(item["p_value"]),
                comment=item["comment"],
            )
            for item in payload["deltas"]
        )
        print(json.dumps(apply_deltas_to_evidence_dto(payload["dto"], deltas)))
        return 0
    catalog = load_write_back_catalog()
    if command == "catalog":
        print(json.dumps(catalog_as_dict(catalog), indent=2))
        return 0
    plans = collect_live_plans(catalog)
    if "--json" in args:
        print(
            json.dumps(
                {
                    "apply_allowed": write_back_is_allowed(catalog.faults),
                    "faults": [asdict(fault) for fault in catalog.faults],
                    "plans": [plan_as_dict(plan) for plan in plans],
                },
                indent=2,
            )
        )
    else:
        print(render_plan(catalog, plans))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
