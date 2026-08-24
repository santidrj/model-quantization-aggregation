"""Effect-specific experimental-unit grouping for relative-improvement aggregation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW

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


def cluster_columns_for_metric(metric: str, paper: Paper) -> list[str] | None:
    """Return grouping columns that identify one cluster unit for ``metric``.

    Cluster units drop evaluation-context columns only. Hardware, library, and
    filter-multiplier factors stay. Artifact-invariant metrics already omit those
    columns from grouping, so their cluster key matches the experimental unit.
    """
    grouping = grouping_columns_for_metric(metric, paper)
    if grouping is None:
        return None
    columns = [column for column in grouping if column not in EVALUATION_CONTEXT_COLUMNS]
    return columns or list(grouping)


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


def cluster_columns_from_unit_columns(unit_columns: list[str]) -> list[str]:
    """Drop evaluation-context columns from an experimental-unit key to form the cluster key."""
    columns = [column for column in unit_columns if column not in EVALUATION_CONTEXT_COLUMNS]
    return columns or list(unit_columns)


def unit_level_statistics(
    values: pl.Series,
    cluster_ids: pl.Series | None = None,
) -> dict[str, float | int | None]:
    """Mean of experimental-unit relative improvements and a 95% interval.

    ``n_eff`` is the number of cluster units when ``cluster_ids`` is provided, otherwise
    the number of experimental units. The interval is a Student's t interval when each
    experimental unit is its own cluster, and a cluster-robust OLS interval when units
    nest inside clusters. The interval is omitted when ``n_eff=1`` or all values are
    identical, because the standard error is then undefined.
    """
    frame = pl.DataFrame({"value": values})
    if cluster_ids is not None:
        frame = frame.with_columns(cluster_ids.alias("cluster"))
    frame = frame.filter(pl.col("value").is_not_null())
    n_units = frame.height
    if n_units == 0:
        return {"n_eff": 0, "mean": None, "lower_ci": None, "upper_ci": None}

    nested = cluster_ids is not None and frame["cluster"].n_unique() < n_units
    n_eff = int(frame["cluster"].n_unique()) if cluster_ids is not None else n_units
    values_arr = frame["value"].to_numpy()
    spread = float(np.ptp(values_arr))
    scale = max(abs(float(np.mean(values_arr))), 1.0)
    if n_units == 1 or spread <= 1e-9 * scale:
        mean = float(values_arr[0]) if n_units == 1 else float(np.mean(values_arr))
        return {"n_eff": n_eff, "mean": mean, "lower_ci": None, "upper_ci": None}

    if not nested:
        stats = DescrStatsW(frame["value"].to_list())
        lower_ci, upper_ci = stats.tconfint_mean(alpha=0.05)
        return {
            "n_eff": n_eff,
            "mean": float(stats.mean),
            "lower_ci": float(lower_ci),
            "upper_ci": float(upper_ci),
        }

    if n_eff < 2:  # noqa: PLR2004
        return {"n_eff": n_eff, "mean": float(frame["value"].mean()), "lower_ci": None, "upper_ci": None}

    response = frame["value"].to_numpy()
    design = np.ones((n_units, 1))
    fit = sm.OLS(response, design).fit(cov_type="cluster", cov_kwds={"groups": frame["cluster"].to_list()})
    lower_ci, upper_ci = fit.conf_int(alpha=0.05)[0]
    return {
        "n_eff": n_eff,
        "mean": float(fit.params[0]),
        "lower_ci": float(lower_ci),
        "upper_ci": float(upper_ci),
    }
