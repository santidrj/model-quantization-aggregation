"""Discount-parameter sensitivity tables for the published analogue."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import polars as pl

from src.belief_assignment import (
    EvidenceModel,
    MassAssignment,
    _effect_name,
    load_evidence_models,
    synthesis_row,
)
from src.belief_discounts import (
    DEFAULT_SATURATION_SIZE,
    DEFAULT_VARIABILITY_CUTOFF,
    DEFAULT_VARIABILITY_K,
    discounted_belief,
    saturation_parameter,
    summarize_effective_sample_sizes,
)
from src.config import PROCESSED_DATA_DIR, TABLES_DIR
from src.data.papers.entities import Papers
from src.dempster_shafer import format_intensity

N0_GRID = (2, "q3", 6)
K_GRID = (0.05, 0.1, 0.2)
CUTOFF_GRID = (3, 4, 8)
ACCURACY_EFFECT = "Accuracy"
ALIZADEH_STUDY_ID = "S14"
DiscountKey = tuple[str, str, str, str]


def load_precision_n_effs(processed_root: Path | None = None) -> list[int]:
    root = processed_root or PROCESSED_DATA_DIR
    frames = [
        pl.read_parquet(path).select("n_eff").filter(pl.col("n_eff").is_not_null())
        for path in sorted(root.glob("*/improvement_statistics_by_precision.parquet"))
    ]
    if not frames:
        return []
    return [int(value) for value in pl.concat(frames)["n_eff"].to_list()]


def load_discount_inputs(
    processed_root: Path | None = None,
) -> dict[DiscountKey, tuple[int, float, float]]:
    """Map (study, method, precision, effect) to (n_eff, iqr, mean)."""
    root = processed_root or PROCESSED_DATA_DIR
    n_eff_by_key: dict[DiscountKey, int] = {}
    for paper in Papers:
        stats_path = root / paper.value.KEY / "improvement_statistics_by_precision.parquet"
        if not stats_path.exists():
            continue
        for row in pl.read_parquet(stats_path).iter_rows(named=True):
            configuration = row["configuration"]
            n_eff_by_key[
                (
                    row["id"],
                    configuration["quantization_method"],
                    configuration["precision_configuration"],
                    row["effect"],
                )
            ] = int(row["n_eff"])

    lookup: dict[DiscountKey, tuple[int, float, float]] = {}
    for paper in Papers:
        effects_path = root / paper.value.KEY / "effects_by_precision.json"
        if not effects_path.exists():
            continue
        for record in json.loads(effects_path.read_text(encoding="utf-8")):
            method = record["quantization_method"]
            precision = record["precision_configuration"]
            for key, payload in record.items():
                if not isinstance(payload, dict) or "iqr" not in payload or "improvement" not in payload:
                    continue
                effect = _effect_name(key)
                n_eff = n_eff_by_key.get((paper.value.ID, method, precision, effect))
                if n_eff is None:
                    continue
                lookup[(paper.value.ID, method, precision, effect)] = (
                    n_eff,
                    float(payload["iqr"]),
                    float(payload["improvement"]),
                )
    return lookup


def remap_evidence_models(
    models: list[EvidenceModel],
    inputs: dict[DiscountKey, tuple[int, float, float]],
    *,
    n0: float,
    k: float,
    cutoff: int,
) -> list[EvidenceModel]:
    remapped: list[EvidenceModel] = []
    for model in models:
        effects: dict[str, tuple[str, float]] = {}
        for effect, (intensity, belief) in model.effects.items():
            key = (model.study_id, model.quantization_method, model.precision_configuration, effect)
            if key not in inputs:
                effects[effect] = (intensity, belief)
                continue
            n_eff, iqr, mean = inputs[key]
            effects[effect] = (
                intensity,
                discounted_belief(model.study_belief, n_eff, iqr, mean, n0=n0, k=k, cutoff=cutoff),
            )
        remapped.append(replace(model, effects=effects))
    return remapped


@dataclass(frozen=True)
class DiscountSensitivityRow:
    label: str
    n0: float
    k: float
    cutoff: int
    is_reference: bool
    accuracy_intensity: str
    accuracy_belief_percent: int
    alizadeh_accuracy_belief: float | None


def _alizadeh_accuracy_belief(models: list[EvidenceModel]) -> float | None:
    preferred: float | None = None
    fallback: float | None = None
    for model in models:
        if model.study_id != ALIZADEH_STUDY_ID or ACCURACY_EFFECT not in model.effects:
            continue
        belief = model.effects[ACCURACY_EFFECT][1]
        fallback = belief if fallback is None else fallback
        if "int4" in model.precision_configuration:
            preferred = belief
    return preferred if preferred is not None else fallback


def discount_sensitivity_rows(
    models: list[EvidenceModel],
    inputs: dict[DiscountKey, tuple[int, float, float]],
    *,
    n0_main: int | None = None,
) -> list[DiscountSensitivityRow]:
    n0_main = DEFAULT_SATURATION_SIZE if n0_main is None else n0_main
    specs: list[tuple[str, float, float, int, bool]] = []
    seen_n0: set[float] = set()
    for item in N0_GRID:
        n0 = float(n0_main if item == "q3" else item)
        if n0 in seen_n0:
            continue
        seen_n0.add(n0)
        specs.append((f"$n_0={n0:g}$", n0, DEFAULT_VARIABILITY_K, DEFAULT_VARIABILITY_CUTOFF, n0 == float(n0_main)))
    for k in K_GRID:
        if k == DEFAULT_VARIABILITY_K:
            continue
        specs.append((f"$k={k}$", float(n0_main), k, DEFAULT_VARIABILITY_CUTOFF, False))
    for cutoff in CUTOFF_GRID:
        if cutoff == DEFAULT_VARIABILITY_CUTOFF:
            continue
        specs.append((f"cutoff ${cutoff}$", float(n0_main), DEFAULT_VARIABILITY_K, cutoff, False))

    rows: list[DiscountSensitivityRow] = []
    for label, n0, k, cutoff, is_reference in specs:
        remapped = remap_evidence_models(models, inputs, n0=n0, k=k, cutoff=cutoff)
        synthesized = synthesis_row(remapped, ACCURACY_EFFECT, MassAssignment.PUBLISHED_ANALOGUE)
        rows.append(
            DiscountSensitivityRow(
                label=label,
                n0=n0,
                k=k,
                cutoff=cutoff,
                is_reference=is_reference,
                accuracy_intensity=format_intensity(synthesized.intensity),
                accuracy_belief_percent=synthesized.belief_percent,
                alizadeh_accuracy_belief=_alizadeh_accuracy_belief(remapped),
            )
        )
    return rows


def render_effective_sample_size_table(n_effs: list[int]) -> str:
    summary = summarize_effective_sample_sizes(n_effs)
    return "\n".join(
        [
            r"\begin{tabular}{ccccccc}",
            r"    \toprule%",
            r"        \rowcolor{gray!30}",
            r"        Mean & Std & Min. & Q1 & Q2 & Q3 & Max. \\",
            r"    \midrule%",
            (
                f"        {summary.mean:.2f} & {summary.std:.2f} & {summary.minimum} & "
                f"{summary.q1:g} & {summary.q2:g} & {summary.q3:g} & {summary.maximum} \\\\"
            ),
            r"    \botrule%",
            r"\end{tabular}",
            "",
        ]
    )


def render_discount_sensitivity_table(rows: list[DiscountSensitivityRow]) -> str:
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Setting & Accuracy intensity & Accuracy belief & S14 accuracy $B'$ \\",
        r"\midrule",
    ]
    for row in rows:
        marker = r"\textbf{reference} " if row.is_reference else ""
        belief = "---" if row.alizadeh_accuracy_belief is None else f"{row.alizadeh_accuracy_belief:.3f}"
        lines.append(
            f"{marker}{row.label} & {row.accuracy_intensity} & {row.accuracy_belief_percent}\\% & {belief} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_discount_sensitivity_tables(
    *,
    processed_root: Path | None = None,
    output_dir: Path | None = None,
    models: list[EvidenceModel] | None = None,
) -> list[Path]:
    directory = output_dir or TABLES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    n_effs = load_precision_n_effs(processed_root)
    n0_main = saturation_parameter(n_effs) if n_effs else DEFAULT_SATURATION_SIZE
    sample_path = directory / "effective-sample-size.tex"
    sample_path.write_text(render_effective_sample_size_table(n_effs or [1]), encoding="utf-8")
    loaded_models = models if models is not None else load_evidence_models(processed_root)
    rows = discount_sensitivity_rows(
        loaded_models,
        load_discount_inputs(processed_root),
        n0_main=n0_main,
    )
    sensitivity_path = directory / "discount-parameter-sensitivity.tex"
    sensitivity_path.write_text(render_discount_sensitivity_table(rows), encoding="utf-8")
    return [sample_path, sensitivity_path]
