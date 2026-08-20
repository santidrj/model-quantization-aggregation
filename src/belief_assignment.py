"""Belief assignment: study-belief masses on evidence models and Dempster–Shafer synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import re

from src.config import PROCESSED_DATA_DIR, TABLES_DIR
from src.data.papers.entities import Papers
from src.data.papers.study_id import study_id_sort_key
from src.dempster_shafer import (
    HypothesisSelectionPolicy,
    combine_effect,
    format_intensity,
    trace_effect,
)


class MassAssignment(Enum):
    PUBLISHED_ANALOGUE = "published_analogue"
    UNDISCOUNTED_UNSPLIT = "undiscounted_unsplit"
    MASS_PRESERVING_BELIEF_SPLIT = "mass_preserving_belief_split"
    EQUITABLE_BELIEF_SPLIT = "equitable_belief_split"


@dataclass(frozen=True)
class EvidenceModel:
    study_id: str
    quantization_method: str
    precision_configuration: str
    study_belief: float
    evidence_model_count: int
    effects: dict[str, tuple[str, float]]
    evidence_factory_id: int | None = None
    aggregation_index: int | None = None


def assigned_mass(model: EvidenceModel, assignment: MassAssignment, effect: str | None = None) -> float:
    if assignment is MassAssignment.EQUITABLE_BELIEF_SPLIT:
        return model.study_belief / model.evidence_model_count
    if assignment is MassAssignment.MASS_PRESERVING_BELIEF_SPLIT:
        return 1.0 - (1.0 - model.study_belief) ** (1.0 / model.evidence_model_count)
    if assignment is MassAssignment.UNDISCOUNTED_UNSPLIT:
        return model.study_belief
    if assignment is MassAssignment.PUBLISHED_ANALOGUE:
        if effect is None:
            raise ValueError("Published analogue mass requires an effect")
        return model.effects[effect][1]
    raise ValueError(f"Unknown mass assignment: {assignment}")


def _model_sort_key(model: EvidenceModel) -> tuple:
    """Evidence Factory combines in aggregation-turn order (Santos 2015, Table 21)."""
    index = model.aggregation_index if model.aggregation_index is not None else 10**18
    return (index, *study_id_sort_key(model.study_id), model.quantization_method, model.precision_configuration)


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


_EVIDENCE_FACTORY_ID_RE = re.compile(r"evidenceEditor/(\d+)")
_STUDY_HEADING_RE = re.compile(r"^## (S\d+)\s*$")
_ORDER_LINE_RE = re.compile(r'^Evidence based on the paper "(?P<title>[^"]+)"(?: \((?P<outer>.+)\))?$')
_TITLE_QUALIFIER_RE = re.compile(r"^(?P<title>.*) \((?P<inner>.+ evidence)\)$")
_AGGREGATION_ORDER_FILENAME = "evidence-factory-aggregation-order.txt"

_TITLE_PREFIX_TO_STUDY = (
    ("Compressing Neural Machine Translation", "S1"),
    ("Impact of Memory Voltage Scaling", "S2"),
    ("Model Quantization and Synthetic Aperture", "S3"),
    ("Benchmarking of Quantization Libraries", "S4"),
    ("Activation Density Based Mixed-Precision", "S5"),
    ("Mixed Precision Low-Bit Quantization", "S6"),
    ("Field Programmable Gate Array-Based All-Layer", "S7"),
    ("Efficient Inference Of Image-Based", "S8"),
    ("Energy-Efficient Respiratory Anomaly", "S9"),
    ("Verifiable and Energy Efficient Medical Image", "S10"),
    ("Experimental Energy Consumption Analysis", "S11"),
    ("Implementing Ultra-Lightweight Co-Inference", "S12"),
    ("Impact of ML Optimization Tactics", "S13"),
    ("Language Models in Software Development", "S14"),
    ("Q_YOLOv5m", "S15"),
    ("POQ: Is There a Pareto-Optimal", "S16"),
    ("Quantized Object Detection", "S17"),
    ("Energy-Efficient Deep Learning for Cloud Detection", "S18"),
    ("Edge AI-Powered System Architecture", "S19"),
    ("Implementing Deep Neural Networks on ARM", "S20"),
    ("Efficient Expiration Date Recognition", "S21"),
)

_FIXED_POINT_WIDTH = {2: "q0.2", 4: "q0.4", 8: "q0.8", 16: "q0.16", 32: "q0.32"}
_TAO_COMPONENT = {"int8": "q0.8", "fp16": "q0.16", "fp32": "fp32", "q0.16": "q0.16", "q0.8": "q0.8"}


def _evidence_factory_ids_by_study(mapping_path: Path) -> dict[str, list[int]]:
    ids_by_study: dict[str, list[int]] = {}
    current_study: str | None = None
    for line in mapping_path.read_text(encoding="utf-8").splitlines():
        heading = _STUDY_HEADING_RE.match(line)
        if heading:
            current_study = heading.group(1)
            ids_by_study.setdefault(current_study, [])
            continue
        match = _EVIDENCE_FACTORY_ID_RE.search(line)
        if match and current_study is not None:
            ids_by_study[current_study].append(int(match.group(1)))
    return ids_by_study


def _parse_aggregation_label(line: str) -> tuple[str, str]:
    match = _ORDER_LINE_RE.match(line.strip())
    if not match:
        raise ValueError(f"Unrecognized aggregation-order line: {line!r}")
    title = match.group("title")
    qualifier = match.group("outer") or ""
    inner = _TITLE_QUALIFIER_RE.match(title)
    if inner:
        title = inner.group("title")
        qualifier = inner.group("inner")
    qualifier = re.sub(r"\s+evidence$", "", qualifier.strip())
    return title, qualifier


def _study_id_for_title(title: str) -> str:
    for prefix, study_id in _TITLE_PREFIX_TO_STUDY:
        if title.startswith(prefix):
            return study_id
    raise ValueError(f"No study mapping for title: {title!r}")


def _compact_wa(qualifier: str) -> str | None:
    compact = re.fullmatch(r"w(\d+)a(\d+)", qualifier.replace(" ", ""), flags=re.IGNORECASE)
    if not compact:
        return None
    weight, activation = compact.group(1), compact.group(2)
    if weight == "16" and activation == "16":
        return "w-fp16, a-fp16"
    return f"w-int{weight}, a-int{activation}"


def _log_bit(qualifier: str) -> list[tuple[str | None, str]]:
    bit = re.fullmatch(r"(\d+)-bit", qualifier.lower())
    return [(None, f"w-log{bit.group(1)}")] if bit else []


def _denkinger_fxp(qualifier: str) -> list[tuple[str | None, str]]:
    fxp = re.fullmatch(r"fxp_(\d+)_(\d+)", qualifier.lower())
    if not fxp:
        return []
    weight, activation = int(fxp.group(1)), int(fxp.group(2))
    return [(None, f"w-{_FIXED_POINT_WIDTH[weight]}, a-{_FIXED_POINT_WIDTH[activation]}")]


def _barnell_format(qualifier: str) -> list[tuple[str | None, str]]:
    lowered = qualifier.lower()
    if lowered == "fp16":
        return [(None, "w-fp16, a-fp16")]
    if lowered == "int8":
        return [(None, "w-int8, a-int8")]
    return []


def _dubhir_format(qualifier: str) -> list[tuple[str | None, str]]:
    lowered = qualifier.lower()
    if lowered.startswith("q"):
        return [(None, f"a-{lowered}")]
    if lowered == "wa-int8":
        return [(None, "w-int8, a-int8")]
    return [(None, qualifier)]


def _xu_method_precision(qualifier: str) -> list[tuple[str | None, str]]:
    method, _, rest = qualifier.partition(" ")
    return [(method, rest)] if method in {"ptq", "qat"} and rest else []


def _paul_bit(qualifier: str) -> list[tuple[str | None, str]]:
    bit = re.fullmatch(r"(\d+)-bit", qualifier.lower())
    if not bit:
        return []
    token = _FIXED_POINT_WIDTH[int(bit.group(1))]
    return [(None, f"w-{token}, a-{token}")]


def _tao_weights_activations(qualifier: str) -> list[tuple[str | None, str]]:
    parts = re.fullmatch(r"weights (.+) - activations (.+)", qualifier.lower())
    if not parts:
        return []
    return [(None, f"w-{_TAO_COMPONENT[parts.group(1)]}, a-{_TAO_COMPONENT[parts.group(2)]}")]


def _alizadeh_bit(qualifier: str) -> list[tuple[str | None, str]]:
    bit = re.fullmatch(r"(\d+)bit", qualifier.lower())
    return [(None, f"w-int{bit.group(1)}")] if bit else []


def _alshammry_wa(qualifier: str) -> list[tuple[str | None, str]]:
    compact = _compact_wa(qualifier.split()[0])
    lowered = qualifier.lower()
    method = "ptq" if "ptq" in lowered else "qat" if "qat" in lowered else None
    return [(method, compact)] if compact and method else []


def _deputter_format(qualifier: str) -> list[tuple[str | None, str]]:
    guesses: list[tuple[str | None, str]] = [(None, qualifier)]
    compact = _compact_wa(qualifier)
    if compact:
        guesses.append((None, compact))
    return guesses


def _guerrouj_format(qualifier: str) -> list[tuple[str | None, str]]:
    lowered = qualifier.lower()
    if lowered == "int8":
        return [(None, "w-int8, a-int8")]
    if lowered == "fp16":
        return [(None, "w-fp16, a-fp16")]
    return []


def _krasteva_format(qualifier: str) -> list[tuple[str | None, str]]:
    compact = _compact_wa(qualifier)
    if compact:
        return [(None, compact)]
    if "full" in qualifier.lower() and "int8" in qualifier.lower():
        return [(None, "full-int8")]
    return []


def _peng_format(qualifier: str) -> list[tuple[str | None, str]]:
    compact = _compact_wa(qualifier)
    return [(None, compact)] if compact else []


_QUALIFIER_RESOLVERS = {
    "S1": _log_bit,
    "S2": _denkinger_fxp,
    "S3": _barnell_format,
    "S4": _dubhir_format,
    "S6": _xu_method_precision,
    "S9": _paul_bit,
    "S11": _tao_weights_activations,
    "S14": _alizadeh_bit,
    "S15": _alshammry_wa,
    "S16": _deputter_format,
    "S17": _guerrouj_format,
    "S20": _krasteva_format,
    "S21": _peng_format,
}


def _candidate_precisions(study_id: str, qualifier: str) -> list[tuple[str | None, str]]:
    """Return (method or None, precision) guesses for a study-specific qualifier."""
    if not qualifier:
        return []
    resolver = _QUALIFIER_RESOLVERS.get(study_id)
    return resolver(qualifier) if resolver else [(None, qualifier)]


def _match_order_label(study_id: str, qualifier: str, study_models: list[EvidenceModel]) -> EvidenceModel:
    if len(study_models) == 1:
        return study_models[0]
    for method, precision in _candidate_precisions(study_id, qualifier):
        matches = [
            model
            for model in study_models
            if model.precision_configuration == precision and (method is None or model.quantization_method == method)
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous {study_id} qualifier {qualifier!r}: {matches}")
    raise ValueError(f"No model for {study_id} qualifier {qualifier!r} among {study_models}")


def _aggregation_indices(models: list[EvidenceModel], order_path: Path) -> dict[tuple[str, str, str], int]:
    by_study: dict[str, list[EvidenceModel]] = {}
    for model in models:
        by_study.setdefault(model.study_id, []).append(model)
    indices: dict[tuple[str, str, str], int] = {}
    lines = [line for line in order_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, line in enumerate(lines):
        title, qualifier = _parse_aggregation_label(line)
        study_id = _study_id_for_title(title)
        matched = _match_order_label(study_id, qualifier, by_study[study_id])
        key = (matched.study_id, matched.quantization_method, matched.precision_configuration)
        if key in indices:
            raise ValueError(f"Duplicate aggregation-order identity {key}")
        indices[key] = index
    if len(indices) != len(models):
        raise ValueError(f"Aggregation order mapped {len(indices)} models, expected {len(models)}")
    return indices


def load_evidence_models(processed_root: Path | None = None) -> list[EvidenceModel]:
    """Load one evidence model per by-precision record in processed JSON."""
    root = processed_root if processed_root is not None else PROCESSED_DATA_DIR
    factory_ids = _evidence_factory_ids_by_study(root / "evidence-diagrams-mapping.md")
    models: list[EvidenceModel] = []
    for paper in Papers:
        path = root / paper.value.KEY / "effects_by_precision.json"
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        evidence_model_count = len(records)
        study_ids = factory_ids.get(paper.value.ID, [])
        for index, record in enumerate(records):
            models.append(
                EvidenceModel(
                    study_id=paper.value.ID,
                    quantization_method=record["quantization_method"],
                    precision_configuration=record["precision_configuration"],
                    study_belief=paper.value.BELIEF,
                    evidence_model_count=evidence_model_count,
                    effects=_effects_from_record(record),
                    evidence_factory_id=study_ids[index] if index < len(study_ids) else None,
                )
            )
    order_path = root / _AGGREGATION_ORDER_FILENAME
    indices = _aggregation_indices(models, order_path)
    return [
        EvidenceModel(
            study_id=model.study_id,
            quantization_method=model.quantization_method,
            precision_configuration=model.precision_configuration,
            study_belief=model.study_belief,
            evidence_model_count=model.evidence_model_count,
            effects=model.effects,
            evidence_factory_id=model.evidence_factory_id,
            aggregation_index=indices[(model.study_id, model.quantization_method, model.precision_configuration)],
        )
        for model in models
    ]


@dataclass(frozen=True)
class PublishedRow:
    effect: str
    intensity: frozenset[str]
    belief_percent: int
    conflict: float
    n_evidence_models: int
    conflict_abs: float


# Last-step K from Evidence Factory aggregated evidence 329074 (not the outdated manuscript transcription).
PUBLISHED_TABLE: tuple[PublishedRow, ...] = (
    PublishedRow("Accuracy", frozenset({"WN", "IF"}), 99, 0.1894921993508596, 41, 0.005),
    PublishedRow("F1 Score", frozenset({"IF"}), 75, 0.14748577564526316, 9, 0.005),
    PublishedRow("mAP", frozenset({"IF"}), 45, 0.31096851862301145, 4, 0.005),
    PublishedRow("Storage Size", frozenset({"SP"}), 100, 6.13678329170885e-10, 62, 5e-10),
    PublishedRow("GPU Utilization", frozenset({"IF"}), 74, 0.0, 3, 1e-12),
    PublishedRow("GPU Power Draw", frozenset({"IF", "WP"}), 98, 0.2958124279750671, 5, 0.005),
    PublishedRow("GPU Energy Consumption", frozenset({"SP"}), 74, 0.11585834464969645, 5, 0.005),
    PublishedRow("RAM Usage", frozenset({"SP"}), 47, 0.19785537468429146, 3, 0.005),
    PublishedRow("Inference Power Draw", frozenset({"WP"}), 72, 0.05798641768183195, 10, 0.005),
    PublishedRow("Inference Energy Consumption", frozenset({"SP"}), 100, 0.003987635134405157, 27, 5e-4),
    PublishedRow("Inference Latency", frozenset({"PO", "SP"}), 100, 0.2692137356332454, 51, 0.005),
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
    *,
    selection_policy: HypothesisSelectionPolicy = HypothesisSelectionPolicy.EVIDENCE_FACTORY_COMPAT,
) -> SynthesisRow:
    """Synthesize one effect; the default preserves Evidence Factory compatibility."""
    pieces = pieces_for_effect(models, effect, assignment)
    if not pieces:
        raise ValueError(f"No evidence models report {effect}")
    combined = combine_effect(pieces, selection_policy=selection_policy)
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
    """Return gate failures for the published analogue vs Evidence Factory 329074."""
    loaded = models if models is not None else load_evidence_models()
    mismatches: list[str] = []
    for expected in PUBLISHED_TABLE:
        actual = synthesis_row(
            loaded,
            expected.effect,
            MassAssignment.PUBLISHED_ANALOGUE,
            selection_policy=HypothesisSelectionPolicy.EVIDENCE_FACTORY_COMPAT,
        )
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
    """Return the four belief assignments under the Santos (2015) selector.

    Evidence Factory literals remain in ``published_*`` fields for impact
    comparison; ``reproduction_mismatches`` is the separate compatibility gate.
    """
    loaded = models if models is not None else load_evidence_models()
    records: list[dict[str, object]] = []
    for expected in PUBLISHED_TABLE:
        local_rows = {
            assignment: synthesis_row(
                loaded,
                expected.effect,
                assignment,
                selection_policy=HypothesisSelectionPolicy.SANTOS_2015,
            )
            for assignment in MassAssignment
        }
        analogue = local_rows[MassAssignment.PUBLISHED_ANALOGUE]
        unsplit = local_rows[MassAssignment.UNDISCOUNTED_UNSPLIT]
        mass_preserving = local_rows[MassAssignment.MASS_PRESERVING_BELIEF_SPLIT]
        equitable = local_rows[MassAssignment.EQUITABLE_BELIEF_SPLIT]
        records.append(
            {
                "effect": expected.effect,
                "selection_policy": HypothesisSelectionPolicy.SANTOS_2015.value,
                "n_evidence_models": analogue.n_evidence_models,
                "analogue_intensity": format_intensity(analogue.intensity),
                "analogue_belief_percent": analogue.belief_percent,
                "analogue_conflict": analogue.conflict,
                "unsplit_intensity": format_intensity(unsplit.intensity),
                "unsplit_belief_percent": unsplit.belief_percent,
                "unsplit_conflict": unsplit.conflict,
                "mass_preserving_intensity": format_intensity(mass_preserving.intensity),
                "mass_preserving_belief_percent": mass_preserving.belief_percent,
                "mass_preserving_conflict": mass_preserving.conflict,
                "equitable_intensity": format_intensity(equitable.intensity),
                "equitable_belief_percent": equitable.belief_percent,
                "equitable_conflict": equitable.conflict,
                "published_intensity": format_intensity(expected.intensity),
                "published_belief_percent": expected.belief_percent,
                "published_conflict": expected.conflict,
            }
        )
    return records


def belief_assignment_trace(
    models: list[EvidenceModel],
    effect: str,
    assignment: MassAssignment,
) -> dict[str, object]:
    """Return ordered provenance and both selector traces for one synthesis."""
    ordered_models = [model for model in sorted(models, key=_model_sort_key) if effect in model.effects]
    if not ordered_models:
        raise ValueError(f"No evidence models report {effect}")
    pieces = [(model.effects[effect][0], assigned_mass(model, assignment, effect=effect)) for model in ordered_models]
    ordered_inputs = [
        {
            "index": index,
            "aggregation_index": model.aggregation_index,
            "study_id": model.study_id,
            "quantization_method": model.quantization_method,
            "precision_configuration": model.precision_configuration,
            "evidence_factory_id": model.evidence_factory_id,
            "intensity_label": label,
            "mass": mass,
        }
        for index, (model, (label, mass)) in enumerate(zip(ordered_models, pieces, strict=True), start=1)
    ]
    traces = {
        policy.value: trace_effect(pieces, selection_policy=policy).to_dict() for policy in HypothesisSelectionPolicy
    }
    compatibility_trace = traces[HypothesisSelectionPolicy.EVIDENCE_FACTORY_COMPAT.value]
    combination = {key: compatibility_trace[key] for key in ("pieces", "steps", "mean_conflict", "final_masses")}
    policies = {
        name: {
            key: trace[key]
            for key in ("selection_policy", "selection_beliefs", "selection_steps", "tie_break", "result")
        }
        for name, trace in traces.items()
    }
    return {
        "effect": effect,
        "assignment": assignment.value,
        "ordered_inputs": ordered_inputs,
        "combination": combination,
        "policies": policies,
    }


_EFFECT_LATEX_NAME = {
    "Accuracy": "Accuracy",
    "F1 Score": r"F$_1$-score",
    "mAP": "mAP",
    "Storage Size": "Storage size",
    "GPU Utilization": "GPU utilization",
    "GPU Power Draw": "GPU power draw",
    "GPU Energy Consumption": "GPU energy",
    "RAM Usage": "RAM usage",
    "Inference Power Draw": "Inf. power draw",
    "Inference Energy Consumption": "Inf. energy",
    "Inference Latency": "Inf. latency",
}

BELIEF_ASSIGNMENT_TABLE_FILENAME = "belief-assignment.tex"


def _latex_intensity(label: str) -> str:
    if label.startswith("{") and label.endswith("}"):
        return r"\{" + label[1:-1] + r"\}"
    return label


def _two_decimals(value: float) -> str:
    return f"{value:.2f}"


def _variant_cells(intensity: str, belief_percent: int, conflict: float) -> list[str]:
    return [
        _latex_intensity(intensity),
        _two_decimals(belief_percent / 100),
        _two_decimals(float(conflict)),
    ]


def render_belief_assignment_table(records: list[dict[str, object]] | None = None) -> str:
    """Render four assignments selected consistently with Santos (2015)."""
    rows = records if records is not None else comparison_records()
    lines = [
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X*{12}{c}}",
        r"\toprule",
        (
            r" & \multicolumn{3}{c}{Published analogue} & \multicolumn{3}{c}{Unsplit}"
            r" & \multicolumn{3}{c}{Mass-preserving} & \multicolumn{3}{c}{Equitable} \\"
        ),
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-13}",
        (
            r"Effect & Intensity & Belief & Conflict & Intensity & Belief & Conflict"
            r" & Intensity & Belief & Conflict & Intensity & Belief & Conflict \\"
        ),
        r"\midrule",
    ]
    for record in rows:
        cells = [_EFFECT_LATEX_NAME.get(str(record["effect"]), str(record["effect"]))]
        for prefix in ("analogue", "unsplit", "mass_preserving", "equitable"):
            cells.extend(
                _variant_cells(
                    str(record[f"{prefix}_intensity"]),
                    int(record[f"{prefix}_belief_percent"]),
                    float(record[f"{prefix}_conflict"]),
                )
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
    return "\n".join(lines)


def write_belief_assignment_table(
    records: list[dict[str, object]] | None = None,
    *,
    output_dir: Path | None = None,
) -> Path:
    """Write the belief-assignment appendix tabular fragment."""
    directory = output_dir if output_dir is not None else TABLES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / BELIEF_ASSIGNMENT_TABLE_FILENAME
    path.write_text(render_belief_assignment_table(records), encoding="utf-8")
    return path
