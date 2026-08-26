"""Single validated synthesis file for manuscript tables, overlays, counts, and prose."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import json
from math import floor, log10
from pathlib import Path
from typing import Any

import polars as pl

from src.belief_assignment import (
    PUBLISHED_TABLE,
    EvidenceModel,
    LeaveOneStudyOutRow,
    MassAssignment,
    SynthesisRow,
    contributing_study_ids,
    leave_one_study_out_records,
    load_evidence_models,
    manuscript_comparison_records,
    mass_preserving_sensitivity_rows,
    synthesis_row,
    write_belief_assignment_table,
    write_leave_one_study_out_table,
    write_sensitivity_mass_preserving_table,
)
from src.config import PROCESSED_DATA_DIR, TABLES_DIR
from src.data.papers.entities import Papers
from src.data.papers.study_id import study_id_sort_key
from src.dempster_shafer import (
    ATOMS,
    HypothesisSelectionPolicy,
    format_intensity,
    intensity_to_hypothesis,
    reconcile_intensities,
)
from src.discount_sensitivity import write_discount_sensitivity_tables
from src.effect_intensity import (
    CorrectnessIntensity,
    CorrectnessMetrics,
    EffectIntensity,
    PerformanceMetrics,
    ResourceEfficiencyMetrics,
    render_intensity_thresholds_table,
)
from src.intensity_threshold_sensitivity import write_intensity_threshold_sensitivity_tables

VALIDATED_SYNTHESIS_FILENAME = "validated-synthesis.json"
VALIDATED_SYNTHESIS_PATH = PROCESSED_DATA_DIR / VALIDATED_SYNTHESIS_FILENAME
SCHEMA_VERSION = 2
MAIN_SELECTION_POLICY = HypothesisSelectionPolicy.EVIDENCE_FACTORY_COMPAT
MAIN_ASSIGNMENT = MassAssignment.PUBLISHED_ANALOGUE
SUBGROUP_BASELINE = "full-fp32"
SUBGROUP_METHOD = "ptq"
SUBGROUP_PRECISION = "w-int8, a-int8"
LEAVE_ONE_OUT_RULE = (
    "For each effect, omit the primary study whose removal yields the smallest remaining aggregated "
    "belief under published-analogue masses and the Evidence Factory-compatible selector. Ties break "
    "toward the omitted study with the lower Study ID. If that remaining belief is not strictly lower "
    "than the full-sample belief, every candidate omission is reported."
)
FORESTPLOT_CLIP = 100
SLIGHTLY_REINFORCED_MAX_PERCENT = 5
HIGH_BELIEF_EMPHASIS_PERCENT = 97
ZERO_CONFLICT = 1e-12
SCIENTIFIC_CONFLICT = 0.01
CONFLICT_ABS = 1e-9
AGGREGATED_EFFECTS_TABLE_FILENAME = "aggregated-effects.tex"
SUBGROUP_TABLE_FILENAME = "subgroup-ptq-w-int8-a-int8.tex"
RESULT_MACROS_FILENAME = "result-macros.tex"
INTENSITY_THRESHOLDS_TABLE_FILENAME = "intensity-thresholds.tex"
PROSE_LOO_FILENAME = "prose-leave-one-study-out.tex"
PROSE_SUBGROUP_INTRO_FILENAME = "prose-subgroup-intro.tex"
PROSE_SUBGROUP_LATENCY_FILENAME = "prose-subgroup-latency.tex"
PROSE_SUBGROUP_NOTE_FILENAME = "prose-subgroup-note.tex"

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

_ATOM_PROSE = {
    "SN": "strongly negatively",
    "NE": "negatively",
    "WN": "weakly negatively",
    "IF": "indifferently",
    "WP": "weakly positively",
    "PO": "positively",
    "SP": "strongly positively",
}

_STUDY_MACRO = {
    "S1": r"\aji",
    "S2": r"\denkinger",
    "S3": r"\barnell",
    "S4": r"\dubhir",
    "S5": r"\vasquez",
    "S6": r"\xu",
    "S7": r"\zhan",
    "S8": r"\flich",
    "S9": r"\paul",
    "S10": r"\sathish",
    "S11": r"\tao",
    "S12": r"\chen",
    "S13": r"\gonzalez",
    "S14": r"\alizadeh",
    "S15": r"\alshammry",
    "S16": r"\deputter",
    "S17": r"\guerrouj",
    "S18": r"\khalil",
    "S19": r"\koli",
    "S20": r"\krasteva",
    "S21": r"\peng",
}

_EFFECT_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Functional Suitability", ("Accuracy", "F1 Score", "mAP")),
    (
        "Resource Efficiency",
        (
            "Storage Size",
            "GPU Utilization",
            "GPU Power Draw",
            "GPU Energy Consumption",
            "RAM Usage",
            "Inference Power Draw",
            "Inference Energy Consumption",
        ),
    ),
    ("Performance", ("Inference Latency",)),
)

_MACRO_STEM = {
    "Accuracy": "Accuracy",
    "F1 Score": "FOne",
    "mAP": "MAP",
    "Storage Size": "Storage",
    "GPU Utilization": "GPUUtil",
    "GPU Power Draw": "GPUPower",
    "GPU Energy Consumption": "GPUEnergy",
    "RAM Usage": "RAMUsage",
    "Inference Power Draw": "InfPower",
    "Inference Energy Consumption": "InfEnergy",
    "Inference Latency": "InfLatency",
}

_CARDINALS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}


def _cardinal(value: int) -> str:
    return _CARDINALS.get(value, str(value))


_CORRECTNESS_EFFECTS = frozenset(CorrectnessMetrics.metrics())


def _ordered_atoms(intensity: Iterable[str]) -> list[str]:
    selected = set(intensity)
    return [atom for atom in ATOMS if atom in selected]


def _sorted_study_ids(study_ids: Iterable[str]) -> list[str]:
    return sorted(study_ids, key=study_id_sort_key)


def _max_contributor_belief(models: Sequence[EvidenceModel], effect: str) -> float:
    return max(model.effects[effect][1] for model in models if effect in model.effects)


def _delta_percent(belief: float, max_contributor_belief: float) -> int:
    return int(round(belief - max_contributor_belief, 2) * 100)


def _delta_class(delta_percent: int) -> str:
    if delta_percent >= SLIGHTLY_REINFORCED_MAX_PERCENT:
        return "reinforced"
    if delta_percent > 0:
        return "slightly_reinforced"
    if delta_percent == 0:
        return "unchanged"
    return "weakened"


def _delta_phrase(delta_percent: int) -> str:
    if delta_percent > 0:
        signed = f"(+{delta_percent}\\%)"
    elif delta_percent < 0:
        signed = f"(--{abs(delta_percent)}\\%)"
    else:
        signed = "(0\\%)"
    label = {
        "reinforced": "Reinforced",
        "slightly_reinforced": "Slightly reinforced",
        "unchanged": "Unchanged",
        "weakened": "Weakened",
    }[_delta_class(delta_percent)]
    return f"{label} {signed}"


def _intensity_prose(intensity: Iterable[str]) -> str:
    ordered = _ordered_atoms(intensity)
    if len(ordered) == 1:
        return _ATOM_PROSE[ordered[0]]
    return " -- ".join(_ATOM_PROSE[atom] for atom in ordered)


def _intensity_reconciliation_effects(
    main_effects: Sequence[SynthesisRow],
    mass_preserving: Sequence[SynthesisRow],
) -> dict[str, Any]:
    """Intersect primary and mass-preserving intensities for theory arrows."""
    check_by_effect = {row.effect: row for row in mass_preserving}
    effects: dict[str, Any] = {}
    for row in main_effects:
        check = check_by_effect[row.effect]
        reconciled = reconcile_intensities(row.intensity, check.intensity)
        differs = row.intensity != check.intensity
        record = {
            "primary_intensity": _ordered_atoms(row.intensity),
            "dependence_check_intensity": _ordered_atoms(check.intensity),
            "differs_from_primary": differs,
        }
        if reconciled is None:
            effects[row.effect] = {
                **record,
                "intensity": None,
                "intensity_label": None,
                "has_theory_arrow": False,
            }
            continue
        effects[row.effect] = {
            **record,
            "intensity": _ordered_atoms(reconciled),
            "intensity_label": format_intensity(reconciled),
            "has_theory_arrow": True,
        }
    return {
        "dependence_check": MassAssignment.MASS_PRESERVING_BELIEF_SPLIT.value,
        "rule": "set_intersection",
        "effects": effects,
    }


def _latex_intensity(label: str) -> str:
    if label.startswith("{") and label.endswith("}"):
        return r"\{" + label[1:-1] + r"\}"
    return label


def _latex_conflict(value: float) -> str:
    if abs(value) < ZERO_CONFLICT:
        return "0"
    if abs(value) < SCIENTIFIC_CONFLICT:
        exponent = floor(log10(abs(value)))
        coefficient = value / (10**exponent)
        return rf"${coefficient:.2f}\times10^{{{exponent}}}$"
    return f"{value:.2f}"


def _latex_belief_percent(value: int) -> str:
    if value >= HIGH_BELIEF_EMPHASIS_PERCENT:
        return rf"\textbf{{{value}\%}}"
    return f"{value}\\%"


def _forestplot_bounds(intensity: Iterable[str], effect: str) -> dict[str, float]:
    ranges = CorrectnessIntensity().get_ranges() if effect in _CORRECTNESS_EFFECTS else EffectIntensity().get_ranges()
    ordered = _ordered_atoms(intensity)
    key = ordered[0] if len(ordered) == 1 else "-".join(ordered)
    if key in ranges:
        lower, upper = ranges[key]
    else:
        lower, upper = ranges[ordered[0]][0], ranges[ordered[-1]][1]
    if lower == float("-inf"):
        lower = -FORESTPLOT_CLIP
    if upper == float("inf"):
        upper = FORESTPLOT_CLIP
    return {"lower_ci": float(lower), "upper_ci": float(upper), "mean": (float(lower) + float(upper)) / 2}


def _effect_payload(
    row: SynthesisRow,
    models: Sequence[EvidenceModel],
    *,
    include_forestplot: bool,
) -> dict[str, Any]:
    max_belief = _max_contributor_belief(models, row.effect)
    delta_percent = _delta_percent(row.belief, max_belief)
    payload: dict[str, Any] = {
        "effect": row.effect,
        "intensity": _ordered_atoms(row.intensity),
        "intensity_label": format_intensity(row.intensity),
        "belief": row.belief,
        "belief_percent": row.belief_percent,
        "conflict": row.conflict,
        "max_contributor_belief": max_belief,
        "delta_belief": row.belief - max_belief,
        "delta_percent": delta_percent,
        "delta_class": _delta_class(delta_percent),
        "n_primary_studies": row.n_primary_studies,
        "n_evidence_models": row.n_evidence_models,
        "study_ids": _sorted_study_ids(contributing_study_ids(list(models), row.effect)),
    }
    if include_forestplot:
        payload["forestplot"] = _forestplot_bounds(row.intensity, row.effect)
    return payload


def _all_effect_names(models: Sequence[EvidenceModel]) -> list[str]:
    names = {effect for model in models for effect in model.effects}
    metric_order = CorrectnessMetrics.metrics() + ResourceEfficiencyMetrics.metrics() + PerformanceMetrics.metrics()
    ordered = [name for name in metric_order if name in names]
    ordered.extend(sorted(names - set(ordered)))
    return ordered


def subgroup_models(models: Sequence[EvidenceModel] | None = None) -> list[EvidenceModel]:
    """Evidence models in the FP32→INT8 PTQ weights-and-activations subgroup."""
    loaded = list(models) if models is not None else load_evidence_models()
    eligible = {paper.value.ID for paper in Papers if paper.value.BASELINE_PRECISION == SUBGROUP_BASELINE}
    return [
        model
        for model in loaded
        if model.study_id in eligible
        and model.quantization_method == SUBGROUP_METHOD
        and model.precision_configuration == SUBGROUP_PRECISION
    ]


def _loo_payload(rows: Sequence[LeaveOneStudyOutRow]) -> dict[str, Any]:
    by_effect: dict[str, list[LeaveOneStudyOutRow]] = defaultdict(list)
    for row in rows:
        by_effect[row.effect].append(row)
    effects: dict[str, Any] = {}
    for expected in PUBLISHED_TABLE:
        group = by_effect[expected.effect]
        reports_all = len(group) > 1
        effects[expected.effect] = {
            "reports_all_omissions": reports_all,
            "omissions": [
                {
                    "omitted_study_id": row.omitted_study_id,
                    "full_intensity": _ordered_atoms(row.full_intensity),
                    "full_belief": row.full_belief,
                    "loo_intensity": _ordered_atoms(row.loo_intensity),
                    "loo_belief": row.loo_belief,
                    "belief_delta": row.belief_delta,
                    "intensity_changed": row.intensity_changed,
                }
                for row in group
            ],
        }
    return {"rule": LEAVE_ONE_OUT_RULE, "effects": effects}


def build_validated_synthesis(models: list[EvidenceModel] | None = None) -> dict[str, Any]:
    """Build the canonical synthesis payload from processed evidence models."""
    loaded = models if models is not None else load_evidence_models()
    main_effects = [
        synthesis_row(loaded, expected.effect, MAIN_ASSIGNMENT, selection_policy=MAIN_SELECTION_POLICY)
        for expected in PUBLISHED_TABLE
    ]
    overlays = [
        synthesis_row(loaded, effect, MAIN_ASSIGNMENT, selection_policy=MAIN_SELECTION_POLICY)
        for effect in _all_effect_names(loaded)
    ]
    subgroup = subgroup_models(loaded)
    subgroup_effect_names = _all_effect_names(subgroup)
    subgroup_rows = [
        synthesis_row(subgroup, effect, MAIN_ASSIGNMENT, selection_policy=MAIN_SELECTION_POLICY)
        for effect in subgroup_effect_names
    ]
    comparison = manuscript_comparison_records(loaded)
    mass_preserving = mass_preserving_sensitivity_rows(loaded, selection_policy=MAIN_SELECTION_POLICY)
    loo_rows = leave_one_study_out_records(loaded, assignment=MAIN_ASSIGNMENT, selection_policy=MAIN_SELECTION_POLICY)

    def _delta(row: SynthesisRow) -> int:
        return _delta_percent(row.belief, _max_contributor_belief(loaded, row.effect))

    increased = [row for row in main_effects if _delta(row) > 0]
    decreased = [row for row in main_effects if _delta(row) < 0]
    unchanged = [row for row in main_effects if _delta(row) == 0]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "selection_policy": MAIN_SELECTION_POLICY.value,
        "assignment": MAIN_ASSIGNMENT.value,
        "corpus": {
            "n_primary_studies": len({model.study_id for model in loaded}),
            "n_evidence_models": len(loaded),
        },
        "counts": {
            "n_increased_belief": len(increased),
            "n_decreased_belief": len(decreased),
            "n_unchanged_belief": len(unchanged),
            "increased_effects": [row.effect for row in increased],
            "decreased_effects": [row.effect for row in decreased],
            "unchanged_effects": [row.effect for row in unchanged],
        },
        "effects": {row.effect: _effect_payload(row, loaded, include_forestplot=True) for row in main_effects},
        "forestplot_overlays": {row.effect: _effect_payload(row, loaded, include_forestplot=True) for row in overlays},
        "belief_assignment": {
            "selection_policy": MAIN_SELECTION_POLICY.value,
            "effects": {
                str(record["effect"]): {
                    "n_primary_studies": int(record["n_primary_studies"]),
                    "n_evidence_models": int(record["n_evidence_models"]),
                    "analogue_intensity": str(record["analogue_intensity"]),
                    "analogue_belief_percent": int(record["analogue_belief_percent"]),
                    "analogue_conflict": float(record["analogue_conflict"]),
                    "unsplit_intensity": str(record["unsplit_intensity"]),
                    "unsplit_belief_percent": int(record["unsplit_belief_percent"]),
                    "unsplit_conflict": float(record["unsplit_conflict"]),
                    "mass_preserving_intensity": str(record["mass_preserving_intensity"]),
                    "mass_preserving_belief_percent": int(record["mass_preserving_belief_percent"]),
                    "mass_preserving_conflict": float(record["mass_preserving_conflict"]),
                    "equitable_intensity": str(record["equitable_intensity"]),
                    "equitable_belief_percent": int(record["equitable_belief_percent"]),
                    "equitable_conflict": float(record["equitable_conflict"]),
                }
                for record in comparison
            },
        },
        "mass_preserving_sensitivity": {
            "selection_policy": MAIN_SELECTION_POLICY.value,
            "effects": {
                row.effect: {
                    "intensity": _ordered_atoms(row.intensity),
                    "intensity_label": format_intensity(row.intensity),
                    "belief": row.belief,
                    "belief_percent": row.belief_percent,
                    "conflict": row.conflict,
                    "n_primary_studies": row.n_primary_studies,
                    "n_evidence_models": row.n_evidence_models,
                }
                for row in mass_preserving
            },
        },
        "intensity_reconciliation": _intensity_reconciliation_effects(main_effects, mass_preserving),
        "leave_one_study_out": _loo_payload(loo_rows),
        "subgroup": {
            "baseline_precision_configuration": SUBGROUP_BASELINE,
            "quantization_method": SUBGROUP_METHOD,
            "precision_configuration": SUBGROUP_PRECISION,
            "n_primary_studies": len({model.study_id for model in subgroup}),
            "n_evidence_models": len(subgroup),
            "one_model_per_study": len(subgroup) == len({model.study_id for model in subgroup}),
            "study_ids": _sorted_study_ids({model.study_id for model in subgroup}),
            "effects": {row.effect: _effect_payload(row, subgroup, include_forestplot=True) for row in subgroup_rows},
        },
    }
    return payload


def validate_synthesis(payload: Mapping[str, Any], models: list[EvidenceModel] | None = None) -> list[str]:  # noqa: PLR0912,PLR0915
    """Return assertion failures for corpus, counts, intensity, belief, conflict, and delta."""
    loaded = models if models is not None else load_evidence_models()
    failures: list[str] = []
    corpus = payload["corpus"]
    if int(corpus["n_evidence_models"]) != len(loaded):
        failures.append(f"corpus n_evidence_models {corpus['n_evidence_models']} != {len(loaded)}")
    n_studies = len({model.study_id for model in loaded})
    if int(corpus["n_primary_studies"]) != n_studies:
        failures.append(f"corpus n_primary_studies {corpus['n_primary_studies']} != {n_studies}")

    for expected in PUBLISHED_TABLE:
        actual = synthesis_row(loaded, expected.effect, MAIN_ASSIGNMENT, selection_policy=MAIN_SELECTION_POLICY)
        stored = payload["effects"][expected.effect]
        if stored["n_primary_studies"] != actual.n_primary_studies:
            failures.append(f"{expected.effect}: studies {stored['n_primary_studies']} != {actual.n_primary_studies}")
        if stored["n_evidence_models"] != actual.n_evidence_models:
            failures.append(f"{expected.effect}: models {stored['n_evidence_models']} != {actual.n_evidence_models}")
        if stored["intensity"] != _ordered_atoms(actual.intensity):
            failures.append(f"{expected.effect}: intensity {stored['intensity']} != {_ordered_atoms(actual.intensity)}")
        if stored["belief_percent"] != actual.belief_percent:
            failures.append(f"{expected.effect}: belief {stored['belief_percent']} != {actual.belief_percent}")
        if abs(float(stored["conflict"]) - actual.conflict) > CONFLICT_ABS:
            failures.append(f"{expected.effect}: conflict {stored['conflict']} != {actual.conflict}")
        expected_delta = _delta_percent(actual.belief, _max_contributor_belief(loaded, expected.effect))
        if stored["delta_percent"] != expected_delta:
            failures.append(f"{expected.effect}: delta {stored['delta_percent']} != {expected_delta}")
        analogue = payload["belief_assignment"]["effects"][expected.effect]
        if analogue["analogue_belief_percent"] != actual.belief_percent:
            failures.append(
                f"{expected.effect}: analogue belief {analogue['analogue_belief_percent']} != {actual.belief_percent}"
            )
        if analogue["analogue_intensity"] != format_intensity(actual.intensity):
            failures.append(
                f"{expected.effect}: analogue intensity {analogue['analogue_intensity']} != "
                f"{format_intensity(actual.intensity)}"
            )
        overlay = payload["forestplot_overlays"][expected.effect]
        if overlay["belief_percent"] != actual.belief_percent:
            failures.append(f"{expected.effect}: overlay belief {overlay['belief_percent']} != {actual.belief_percent}")
        if overlay["intensity"] != _ordered_atoms(actual.intensity):
            failures.append(f"{expected.effect}: overlay intensity {overlay['intensity']} != actual")
        mass = payload["mass_preserving_sensitivity"]["effects"][expected.effect]
        comparison_mass = analogue["mass_preserving_belief_percent"]
        if mass["belief_percent"] != comparison_mass:
            failures.append(
                f"{expected.effect}: mass-preserving {mass['belief_percent']} != appendix {comparison_mass}"
            )
        recon = payload["intensity_reconciliation"]["effects"][expected.effect]
        check_intensity = frozenset(mass["intensity"])
        expected_recon = reconcile_intensities(actual.intensity, check_intensity)
        if expected_recon is None:
            if recon["intensity"] is not None or recon["has_theory_arrow"]:
                failures.append(f"{expected.effect}: reconciliation should omit theory arrow")
        else:
            if recon["intensity"] != _ordered_atoms(expected_recon):
                failures.append(
                    f"{expected.effect}: reconciled intensity {recon['intensity']} != {_ordered_atoms(expected_recon)}"
                )
            if not recon["has_theory_arrow"]:
                failures.append(f"{expected.effect}: reconciliation should keep theory arrow")
        if bool(recon["differs_from_primary"]) != (actual.intensity != check_intensity):
            failures.append(f"{expected.effect}: differs_from_primary mismatch")

    counts = payload["counts"]
    increased = [name for name, row in payload["effects"].items() if int(row["delta_percent"]) > 0]
    decreased = [name for name, row in payload["effects"].items() if int(row["delta_percent"]) < 0]
    if counts["increased_effects"] != increased:
        failures.append(f"increased effects {counts['increased_effects']} != {increased}")
    if counts["decreased_effects"] != decreased:
        failures.append(f"decreased effects {counts['decreased_effects']} != {decreased}")
    if int(counts["n_increased_belief"]) != len(increased):
        failures.append("n_increased_belief does not match positive deltas")
    if int(counts["n_decreased_belief"]) != len(decreased):
        failures.append("n_decreased_belief does not match negative deltas")

    subgroup = payload["subgroup"]
    actual_subgroup = subgroup_models(loaded)
    if int(subgroup["n_primary_studies"]) != len({model.study_id for model in actual_subgroup}):
        failures.append("subgroup study count mismatch")
    if int(subgroup["n_evidence_models"]) != len(actual_subgroup):
        failures.append("subgroup evidence-model count mismatch")
    if subgroup["study_ids"] != _sorted_study_ids({model.study_id for model in actual_subgroup}):
        failures.append("subgroup study_ids mismatch")
    for effect, stored in subgroup["effects"].items():
        if int(stored["n_evidence_models"]) != int(stored["n_primary_studies"]) and subgroup["one_model_per_study"]:
            failures.append(f"subgroup {effect}: model count {stored['n_evidence_models']} exceeds one-per-study")

    for effect, loo in payload["leave_one_study_out"]["effects"].items():
        omissions = loo["omissions"]
        if not omissions:
            failures.append(f"{effect}: missing leave-one-study-out omissions")
            continue
        if loo["reports_all_omissions"]:
            if any(float(item["loo_belief"]) < float(item["full_belief"]) for item in omissions):
                failures.append(f"{effect}: reported all omissions despite a belief decrease")
        elif float(omissions[0]["loo_belief"]) >= float(omissions[0]["full_belief"]):
            failures.append(f"{effect}: selected omission does not decrease belief")
    return failures


def assert_validated_synthesis(payload: Mapping[str, Any], models: list[EvidenceModel] | None = None) -> None:
    failures = validate_synthesis(payload, models)
    if failures:
        raise AssertionError("Validated synthesis assertions failed:\n" + "\n".join(failures))


def write_validated_synthesis(
    payload: Mapping[str, Any] | None = None,
    *,
    path: Path | None = None,
    models: list[EvidenceModel] | None = None,
) -> Path:
    built = dict(payload) if payload is not None else build_validated_synthesis(models)
    assert_validated_synthesis(built, models)
    target = path if path is not None else VALIDATED_SYNTHESIS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(built, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target


def load_validated_synthesis(path: Path | None = None) -> dict[str, Any]:
    target = path if path is not None else VALIDATED_SYNTHESIS_PATH
    return json.loads(target.read_text(encoding="utf-8"))


def forestplot_overlay_frame(
    payload: Mapping[str, Any] | None = None,
    *,
    key: str = "forestplot_overlays",
) -> pl.DataFrame:
    """Aggregated forest-plot rows generated from the validated synthesis file.

    ``lower_ci``/``upper_ci`` are SSM intensity-range endpoints on the relative-improvement
    axis, not sampling intervals. Unbounded SN and SP ranges are clipped to
    ``FORESTPLOT_CLIP``.
    """
    data = payload if payload is not None else load_validated_synthesis()
    rows = []
    for effect, record in data[key].items():
        bounds = record["forestplot"]
        rows.append(
            {
                "effect": effect,
                "lower_ci": int(bounds["lower_ci"]),
                "upper_ci": int(bounds["upper_ci"]),
                "belief": round(record["belief"], 3),
                "mean": bounds["mean"],
                "n_eff": None,
                "id": "Aggregated",
                "evidence_label": "Aggregated",
            }
        )
    return pl.from_dicts(rows)


def subgroup_forestplot_overlay_frame(payload: Mapping[str, Any] | None = None) -> pl.DataFrame:
    data = payload if payload is not None else load_validated_synthesis()
    frame = forestplot_overlay_frame({"forestplot_overlays": data["subgroup"]["effects"]})
    return frame.with_columns(
        pl.lit(SUBGROUP_METHOD).alias("quantization_method"),
        pl.lit(SUBGROUP_PRECISION).alias("precision_configuration"),
        pl.lit("e0").alias("evidence_id"),
    )


def render_aggregated_effects_table(payload: Mapping[str, Any] | None = None) -> str:
    data = payload if payload is not None else load_validated_synthesis()
    lines = [
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X",
        r"  >{\centering\arraybackslash}m{2cm}",
        r"  >{\centering\arraybackslash}m{1cm}",
        r"  >{\centering\arraybackslash}m{1.4cm}",
        r"  >{\centering\arraybackslash}X",
        r"  >{\centering\arraybackslash}m{0.9cm}",
        r"  >{\centering\arraybackslash}m{0.9cm}}",
        r"\toprule",
        r"\rowcolor{gray!30}%",
        (
            r"\textbf{Effect} & \textbf{Direction \& intensity}\footnotemark[1] & \textbf{Belief} & "
            r"\textbf{Conflict} & \textbf{$\Delta$ Belief} & \textbf{Studies} & \textbf{Models} \\"
        ),
        r"\midrule",
    ]
    for group_label, effect_names in _EFFECT_GROUPS:
        lines.append(r"\rowcolor{gray!20}%")
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{group_label}}}}} \\")
        for effect in effect_names:
            row = data["effects"][effect]
            lines.append(
                " & ".join(
                    [
                        _EFFECT_LATEX_NAME[effect],
                        _latex_intensity(row["intensity_label"]),
                        _latex_belief_percent(int(row["belief_percent"])),
                        _latex_conflict(float(row["conflict"])),
                        _delta_phrase(int(row["delta_percent"])),
                        str(row["n_primary_studies"]),
                        str(row["n_evidence_models"]),
                    ]
                )
                + r" \\"
            )
        if group_label != _EFFECT_GROUPS[-1][0]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
    return "\n".join(lines)


def render_subgroup_table(payload: Mapping[str, Any] | None = None) -> str:
    data = payload if payload is not None else load_validated_synthesis()
    effects = data["subgroup"]["effects"]
    lines = [
        r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X",
        r"  >{\centering\arraybackslash}m{2cm}",
        r"  >{\centering\arraybackslash}m{1cm}",
        r"  >{\centering\arraybackslash}m{1.4cm}",
        r"  >{\centering\arraybackslash}X",
        r"  >{\centering\arraybackslash}m{0.9cm}",
        r"  >{\centering\arraybackslash}m{0.9cm}}",
        r"\toprule",
        r"\rowcolor{gray!30}%",
        (
            r"\textbf{Effect} & \textbf{Direction \& intensity}\footnotemark[1] & \textbf{Belief} & "
            r"\textbf{Conflict} & \textbf{$\Delta$ Belief} & \textbf{Studies} & \textbf{Models} \\"
        ),
        r"\midrule",
    ]
    for group_label, effect_names in _EFFECT_GROUPS:
        present = [effect for effect in effect_names if effect in effects]
        if not present:
            continue
        lines.append(r"\rowcolor{gray!20}%")
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{group_label}}}}} \\")
        for effect in present:
            row = effects[effect]
            lines.append(
                " & ".join(
                    [
                        _EFFECT_LATEX_NAME[effect],
                        _latex_intensity(row["intensity_label"]),
                        f"{int(row['belief_percent'])}\\%",
                        _latex_conflict(float(row["conflict"])),
                        _delta_phrase(int(row["delta_percent"])),
                        str(row["n_primary_studies"]),
                        str(row["n_evidence_models"]),
                    ]
                )
                + r" \\"
            )
        if group_label != _EFFECT_GROUPS[-1][0]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
    return "\n".join(lines)


def _study_macros(study_ids: Sequence[str]) -> str:
    return ", ".join(_STUDY_MACRO[study_id] for study_id in study_ids)


def render_result_macros(payload: Mapping[str, Any] | None = None) -> str:
    data = payload if payload is not None else load_validated_synthesis()
    lines = [
        r"% Generated from data/processed/validated-synthesis.json. Do not edit by hand.",
        rf"\newcommand{{\AggNIncreased}}{{{data['counts']['n_increased_belief']}}}",
        rf"\newcommand{{\AggNDecreased}}{{{data['counts']['n_decreased_belief']}}}",
        rf"\newcommand{{\AggNUnchanged}}{{{data['counts']['n_unchanged_belief']}}}",
        rf"\newcommand{{\AggCorpusStudies}}{{{data['corpus']['n_primary_studies']}}}",
        rf"\newcommand{{\AggCorpusModels}}{{{data['corpus']['n_evidence_models']}}}",
    ]
    for effect, row in data["effects"].items():
        stem = _MACRO_STEM[effect]
        lines.append(rf"\newcommand{{\Agg{stem}Belief}}{{{int(row['belief_percent'])}\%}}")
        lines.append(rf"\newcommand{{\Agg{stem}Intensity}}{{{_latex_intensity(row['intensity_label'])}}}")
        delta = int(row["delta_percent"])
        delta_tex = f"+{delta}\\%" if delta > 0 else f"--{abs(delta)}\\%"
        lines.append(rf"\newcommand{{\Agg{stem}Delta}}{{{delta_tex}}}")
        lines.append(rf"\newcommand{{\Agg{stem}Studies}}{{{row['n_primary_studies']}}}")
        lines.append(rf"\newcommand{{\Agg{stem}Models}}{{{row['n_evidence_models']}}}")
        lines.append(rf"\newcommand{{\Agg{stem}Prose}}{{{_intensity_prose(row['intensity'])}}}")
    mass = data["mass_preserving_sensitivity"]["effects"]
    lines.append(rf"\newcommand{{\MassPresAccuracyBelief}}{{{int(mass['Accuracy']['belief_percent'])}\%}}")
    lines.append(
        rf"\newcommand{{\MassPresAccuracyIntensity}}{{{_latex_intensity(mass['Accuracy']['intensity_label'])}}}"
    )
    lines.append(rf"\newcommand{{\MassPresLatencyBelief}}{{{int(mass['Inference Latency']['belief_percent'])}\%}}")
    lines.append(
        rf"\newcommand{{\MassPresLatencyIntensity}}{{{_latex_intensity(mass['Inference Latency']['intensity_label'])}}}"
    )
    recon = data["intensity_reconciliation"]["effects"]
    for effect, stem in _MACRO_STEM.items():
        record = recon[effect]
        has_arrow = "true" if record["has_theory_arrow"] else "false"
        lines.append(rf"\newcommand{{\Theory{stem}HasArrow}}{{{has_arrow}}}")
        if record["has_theory_arrow"]:
            lines.append(rf"\newcommand{{\Theory{stem}Intensity}}{{{_latex_intensity(record['intensity_label'])}}}")
            lines.append(rf"\newcommand{{\Theory{stem}Prose}}{{{_intensity_prose(record['intensity'])}}}")
        else:
            lines.append(rf"\newcommand{{\Theory{stem}Intensity}}{{}}")
            lines.append(rf"\newcommand{{\Theory{stem}Prose}}{{}}")
    subgroup = data["subgroup"]
    lines.append(rf"\newcommand{{\SubgroupNStudies}}{{{subgroup['n_primary_studies']}}}")
    lines.append(rf"\newcommand{{\SubgroupNModels}}{{{subgroup['n_evidence_models']}}}")
    lines.append(rf"\newcommand{{\SubgroupSources}}{{{_study_macros(subgroup['study_ids'])}}}")
    latency = subgroup["effects"]["Inference Latency"]
    lines.append(rf"\newcommand{{\SubgroupLatencyNStudies}}{{{latency['n_primary_studies']}}}")
    lines.append(rf"\newcommand{{\SubgroupLatencyNModels}}{{{latency['n_evidence_models']}}}")
    lines.append(
        rf"\newcommand{{\SubgroupMAPIntensity}}{{{_latex_intensity(subgroup['effects']['mAP']['intensity_label'])}}}"
    )
    lines.append(rf"\newcommand{{\SubgroupLatencyIntensity}}{{{_latex_intensity(latency['intensity_label'])}}}")
    loo = data["leave_one_study_out"]["effects"]
    storage = loo["Storage Size"]["omissions"][0]
    latency_loo = loo["Inference Latency"]["omissions"][0]
    accuracy_loo = loo["Accuracy"]["omissions"][0]
    lines.append(rf"\newcommand{{\LOOStorageOmitted}}{{{storage['omitted_study_id']}}}")
    lines.append(rf"\newcommand{{\LOOLatencyOmitted}}{{{latency_loo['omitted_study_id']}}}")
    lines.append(rf"\newcommand{{\LOOLatencyFull}}{{{float(latency_loo['full_belief']):.2f}}}")
    lines.append(rf"\newcommand{{\LOOLatencyLOO}}{{{float(latency_loo['loo_belief']):.2f}}}")
    lines.append(rf"\newcommand{{\LOOAccuracyOmitted}}{{{accuracy_loo['omitted_study_id']}}}")
    lines.append("")
    return "\n".join(lines)


def render_loo_prose(payload: Mapping[str, Any] | None = None) -> str:
    data = payload if payload is not None else load_validated_synthesis()
    loo = data["leave_one_study_out"]
    all_report = [effect for effect, record in loo["effects"].items() if record["reports_all_omissions"]]
    storage = loo["effects"]["Storage Size"]["omissions"][0]
    latency = loo["effects"]["Inference Latency"]["omissions"][0]
    accuracy = loo["effects"]["Accuracy"]["omissions"][0]
    extra = ""
    if all_report:
        clauses = []
        for effect in all_report:
            omissions = loo["effects"][effect]["omissions"]
            listed = ", ".join(f"{item['omitted_study_id']} ({float(item['belief_delta']):+.2f})" for item in omissions)
            clauses.append(f"{_EFFECT_LATEX_NAME[effect]} increases under every omission ({listed})")
        extra = " " + "; ".join(clauses) + "."
    return (
        rf"Table~\ref{{tab:leave-one-study-out}} recomputes the main discounted pooling under the following "
        rf"rule. {LEAVE_ONE_OUT_RULE} Intensities marked with \textsuperscript{{*}} change under that "
        rf"omission. Storage size is unchanged when the largest contributing study "
        rf"({storage['omitted_study_id']}, 17 evidence models) is removed, whereas inference latency drops "
        rf"from belief ${float(latency['full_belief']):.2f}$ to ${float(latency['loo_belief']):.2f}$ under "
        rf"the same omission while retaining intensity "
        rf"{_latex_intensity(format_intensity(frozenset(latency['full_intensity'])))}. Accuracy's largest "
        rf"drop arises from omitting {accuracy['omitted_study_id']} (one evidence model), illustrating that "
        rf"high belief is not explained solely by configuration multiplicity.{extra}"
        "\n"
    )


def render_subgroup_intro(payload: Mapping[str, Any] | None = None) -> str:
    data = payload if payload is not None else load_validated_synthesis()
    subgroup = data["subgroup"]
    each = ", each contributing a single evidence model" if subgroup["one_model_per_study"] else ""
    return (
        "To illustrate the impact of configuration alignment on aggregation outcomes, we conducted a "
        "subgroup analysis focusing on a single, widely used configuration: post-training quantization of "
        "weights and activations from FP32 to INT8. This subgroup includes "
        rf"{_cardinal(int(subgroup['n_primary_studies']))} studies{each}, in contrast to the "
        "multi-evidence contributions "
        r"used in the full aggregation from Section~\ref{sec:results}. Table~\ref{tab:subgroup-analysis-results} "
        "summarizes the resulting effects.\n"
    )


def render_subgroup_note(payload: Mapping[str, Any] | None = None) -> str:
    data = payload if payload is not None else load_validated_synthesis()
    sources = _study_macros(data["subgroup"]["study_ids"])
    return rf"\textbf{{Note:}} Evidence sources for this subgroup include {sources}." + "\n"


def render_subgroup_latency_prose(payload: Mapping[str, Any] | None = None) -> str:
    data = payload if payload is not None else load_validated_synthesis()
    models = subgroup_models()
    below_sp = []
    for model in sorted(models, key=lambda item: study_id_sort_key(item.study_id)):
        if "Inference Latency" not in model.effects:
            continue
        intensity = intensity_to_hypothesis(model.effects["Inference Latency"][0])
        if intensity != frozenset({"SP"}):
            below_sp.append(f"{_STUDY_MACRO[model.study_id]} reports {_latex_intensity(format_intensity(intensity))}")
    latency = data["subgroup"]["effects"]["Inference Latency"]
    listed = "; ".join(below_sp)
    return (
        "Second, the effect on inference latency shifts from "
        r"\emph{\AggInfLatencyProse} (\AggInfLatencyIntensity) in the full aggregation to a "
        r"\emph{strongly positive} (\SubgroupLatencyIntensity) effect in the subgroup analysis. Among the "
        rf"{_cardinal(int(latency['n_primary_studies']))} studies reporting this effect, "
        rf"intensities other than singleton SP "
        rf"are: {listed}."
        "\n"
    )


def write_validated_tables(
    payload: Mapping[str, Any] | None = None,
    *,
    output_dir: Path | None = None,
) -> list[Path]:
    data = dict(payload) if payload is not None else build_validated_synthesis()
    assert_validated_synthesis(data)
    directory = output_dir if output_dir is not None else TABLES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    written = [
        write_belief_assignment_table(manuscript_comparison_records(), output_dir=directory),
        write_leave_one_study_out_table(output_dir=directory),
        write_sensitivity_mass_preserving_table(output_dir=directory),
    ]
    fragments = {
        AGGREGATED_EFFECTS_TABLE_FILENAME: render_aggregated_effects_table(data),
        SUBGROUP_TABLE_FILENAME: render_subgroup_table(data),
        RESULT_MACROS_FILENAME: render_result_macros(data),
        PROSE_LOO_FILENAME: render_loo_prose(data),
        PROSE_SUBGROUP_INTRO_FILENAME: render_subgroup_intro(data),
        PROSE_SUBGROUP_NOTE_FILENAME: render_subgroup_note(data),
        PROSE_SUBGROUP_LATENCY_FILENAME: render_subgroup_latency_prose(data),
        INTENSITY_THRESHOLDS_TABLE_FILENAME: render_intensity_thresholds_table(),
    }
    for filename, text in fragments.items():
        path = directory / filename
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


def write_all_validated_outputs(
    *,
    models: list[EvidenceModel] | None = None,
    json_path: Path | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Write the validated JSON and every manuscript fragment derived from it."""
    payload = build_validated_synthesis(models)
    json_file = write_validated_synthesis(payload, path=json_path, models=models)
    table_paths = write_validated_tables(payload, output_dir=output_dir)
    sensitivity_paths = write_discount_sensitivity_tables(output_dir=output_dir, models=models)
    threshold_paths = write_intensity_threshold_sensitivity_tables(output_dir=output_dir, models=models)
    return [json_file, *table_paths, *sensitivity_paths, *threshold_paths]
