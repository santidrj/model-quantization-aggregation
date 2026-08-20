"""Effect-specific experimental-unit grouping for relative-improvement aggregation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from statsmodels.stats import descriptivestats as sms

if TYPE_CHECKING:
    from src.data.papers.entities import Paper

EVALUATION_CONTEXT_COLUMNS = frozenset({"Dataset", "dataset", "task"})
ARTIFACT_INVARIANT_METRICS = frozenset({"storage_size"})
PRECISION_KEY_COLUMNS = ("quantization_method", "precision_configuration")


def grouping_columns_for_metric(metric: str, paper: Paper) -> list[str] | None:
    """Return grouping columns that define one experimental unit for ``metric``."""
    overrides = getattr(paper, "METRIC_GROUPING_OVERRIDES", None) or {}
    if metric in overrides:
        return list(overrides[metric])

    if paper.GROUPING_COLUMNS is None:
        return None

    if metric in ARTIFACT_INVARIANT_METRICS:
        cols = [column for column in paper.GROUPING_COLUMNS if column not in EVALUATION_CONTEXT_COLUMNS]
        return cols or list(paper.GROUPING_COLUMNS)

    return list(paper.GROUPING_COLUMNS)


def unit_columns_for_precision(metric: str, paper: Paper) -> list[str]:
    """Columns that identify one experimental unit inside a by-precision aggregation."""
    grouping = grouping_columns_for_metric(metric, paper)
    if grouping is None:
        return list(PRECISION_KEY_COLUMNS)
    return list(dict.fromkeys([*grouping, *PRECISION_KEY_COLUMNS]))


def unit_columns_for_configuration(metric: str, paper: Paper, available_columns: set[str]) -> list[str]:
    """Columns that identify one experimental unit inside a configuration group."""
    grouping = grouping_columns_for_metric(metric, paper) or paper.GROUPING_COLUMNS or []
    columns = ["configuration"]
    for column in grouping:
        if column in available_columns:
            columns.append(column)
    return list(dict.fromkeys(columns))


def collapse_metric_to_units(
    improvement_metrics: pl.DataFrame,
    metric: str,
    unit_columns: list[str],
) -> pl.DataFrame:
    """Average replicate rows to one relative improvement per experimental unit."""
    improvement_column = f"{metric}_improvement"
    return improvement_metrics.group_by(unit_columns).agg(pl.col(improvement_column).mean().alias(improvement_column))


def unit_level_statistics(values: pl.Series) -> dict[str, float | int | None]:
    """Mean and 95% CI from unit-level relative improvements."""
    clean = values.drop_nulls()
    n_eff = clean.len()
    if n_eff == 0:
        return {"n_eff": 0, "mean": None, "lower_ci": None, "upper_ci": None}
    if n_eff == 1 or clean.n_unique() == 1:
        mean = clean.item(0) if n_eff == 1 else clean.unique().item()
        return {"n_eff": n_eff, "mean": mean, "lower_ci": None, "upper_ci": None}

    frame = pl.DataFrame({"value": clean})
    stats = pl.from_pandas(sms.describe(frame, stats=["mean", "ci"], alpha=0.05).T)
    return {
        "n_eff": n_eff,
        "mean": stats["mean"].item(),
        "lower_ci": stats["lower_ci"].item(),
        "upper_ci": stats["upper_ci"].item(),
    }
