"""How the pooled accuracy intensity arises from its evidence models.

Descriptive breakdown of the accuracy evidence models (assigned intensity, mean
relative improvement, contributing studies) next to the combined Dempster--Shafer
masses, so the manuscript can show why the pooled result is {I} and what its
belief does and does not say. Uses only existing extraction outputs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from statistics import median
from typing import Any

import polars as pl

from src.belief_assignment import EvidenceModel, MassAssignment, pieces_for_effect
from src.config import PROCESSED_DATA_DIR
from src.data.papers.entities import Papers
from src.data.papers.study_id import study_id_sort_key
from src.dempster_shafer import (
    ATOMS,
    HypothesisSelectionPolicy,
    belief,
    format_intensity,
    intensity_to_hypothesis,
    trace_effect,
)

ACCURACY_EFFECT = "Accuracy"
MODERATE_LOSS_PERCENT = -5.0
"""Mean RI at or below this value lies in the {WN} band or worse for functional suitability."""
STRONG_LOSS_PERCENT = -25.0
"""Mean RI below this value lies in the {SN} band for functional suitability."""

_ModelKey = tuple[str, str, str]


def _ordered(hypothesis: frozenset[str]) -> list[str]:
    return [atom for atom in ATOMS if atom in hypothesis]


def load_mean_relative_improvements(
    effect: str = ACCURACY_EFFECT,
    processed_root: Path | None = None,
) -> dict[_ModelKey, float]:
    """Map (study, method, precision) to the evidence model's mean RI (%) for ``effect``."""
    root = processed_root or PROCESSED_DATA_DIR
    means: dict[_ModelKey, float] = {}
    for paper in Papers:
        path = root / paper.value.KEY / "improvement_statistics_by_precision.parquet"
        if not path.exists():
            continue
        for row in pl.read_parquet(path).filter(pl.col("effect") == effect).iter_rows(named=True):
            configuration = row["configuration"]
            key = (row["id"], configuration["quantization_method"], configuration["precision_configuration"])
            means[key] = float(row["mean"])
    return means


def accuracy_breakdown(
    models: list[EvidenceModel],
    *,
    assignment: MassAssignment,
    selection_policy: HypothesisSelectionPolicy,
    mean_relative_improvements: Mapping[_ModelKey, float] | None = None,
) -> dict[str, Any]:
    """Summarize accuracy evidence models by assigned intensity and their combined masses."""
    contributing = [model for model in models if ACCURACY_EFFECT in model.effects]
    means = mean_relative_improvements if mean_relative_improvements is not None else load_mean_relative_improvements()
    model_means = [
        means[(model.study_id, model.quantization_method, model.precision_configuration)] for model in contributing
    ]

    trace = trace_effect(
        pieces_for_effect(models, ACCURACY_EFFECT, assignment),
        selection_policy=selection_policy,
    )
    final_masses = {frozenset(entry.hypothesis): entry.mass for entry in trace.final_masses}

    by_intensity: dict[frozenset[str], list[int]] = defaultdict(list)
    for index, model in enumerate(contributing):
        by_intensity[intensity_to_hypothesis(model.effects[ACCURACY_EFFECT][0])].append(index)

    def sort_key(hypothesis: frozenset[str]) -> tuple[int, int]:
        return (min(ATOMS.index(atom) for atom in hypothesis), len(hypothesis))

    rows = []
    for hypothesis in sorted(by_intensity, key=sort_key):
        group = by_intensity[hypothesis]
        group_means = [model_means[index] for index in group]
        rows.append(
            {
                "intensity": _ordered(hypothesis),
                "intensity_label": format_intensity(hypothesis),
                "n_evidence_models": len(group),
                "study_ids": sorted({contributing[index].study_id for index in group}, key=study_id_sort_key),
                "min_mean_ri": min(group_means),
                "max_mean_ri": max(group_means),
                "combined_mass": final_masses.get(hypothesis, 0.0),
            }
        )

    all_means = model_means
    strong_loss = [contributing[index] for index, value in enumerate(model_means) if value < STRONG_LOSS_PERCENT]
    negative_atoms = frozenset({"SN", "N", "WN"})
    below_weak = frozenset({"SN", "N"})
    conflicts = [step.conflict for step in trace.steps]
    return {
        "effect": ACCURACY_EFFECT,
        "n_evidence_models": len(contributing),
        "n_primary_studies": len({model.study_id for model in contributing}),
        "rows": rows,
        "n_indifferent_only": len(by_intensity.get(frozenset({"I"}), [])),
        "n_including_indifferent": sum(len(group) for hyp, group in by_intensity.items() if "I" in hyp),
        "n_strictly_negative": sum(len(group) for hyp, group in by_intensity.items() if hyp <= negative_atoms),
        "n_positive_intensity": sum(
            len(group) for hyp, group in by_intensity.items() if hyp & frozenset({"WP", "P", "SP"})
        ),
        "n_negative_mean": sum(value < 0 for value in all_means),
        "n_moderate_loss": sum(value <= MODERATE_LOSS_PERCENT for value in all_means),
        "n_strong_loss": len(strong_loss),
        "strong_loss_study_ids": sorted({model.study_id for model in strong_loss}, key=study_id_sort_key),
        "median_mean_ri": median(all_means),
        "min_mean_ri": min(all_means),
        "selected_intensity": _ordered(trace.result.intensity),
        "belief_indifferent": belief(final_masses, frozenset({"I"})),
        "belief_weakly_negative": belief(final_masses, frozenset({"WN"})),
        "belief_weakly_negative_or_indifferent": belief(final_masses, frozenset({"WN", "I"})),
        "belief_below_weakly_negative": belief(final_masses, below_weak | frozenset({"WN"}))
        - belief(final_masses, frozenset({"WN"})),
        "max_pairwise_conflict": max(conflicts) if conflicts else 0.0,
        "last_pairwise_conflict": conflicts[-1] if conflicts else 0.0,
        "n_combination_steps": len(conflicts),
    }
