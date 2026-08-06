"""Sensitivity analysis helpers for study-set restrictions of the main by-precision aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.data.papers.study_id import study_id_rank_expr

SYNTHETIC_STAT_IDS: frozenset[str] = frozenset({"Summary", "Aggregated"})


def _primary_study_rows(stats_df: pl.DataFrame) -> pl.DataFrame:
    return stats_df.filter(~pl.col("id").is_in(list(SYNTHETIC_STAT_IDS)))


def ts_counts_by_study(stats_df: pl.DataFrame) -> pl.DataFrame:
    """Per-study theoretical-structure (TS) count: unique ``evidence_id`` among primary-study rows."""
    return (
        _primary_study_rows(stats_df)
        .group_by("id")
        .agg(pl.col("evidence_id").n_unique().alias("ts_count"))
        .sort(study_id_rank_expr())
    )


def mark_included_by_ts_count(ts_counts: pl.DataFrame, *, max_ts: int = 5) -> pl.DataFrame:
    """Add ``included`` where ``ts_count <= max_ts`` (inclusive)."""
    if max_ts < 1:
        raise ValueError(f"max_ts must be >= 1, got {max_ts}")
    return ts_counts.with_columns((pl.col("ts_count") <= max_ts).alias("included"))


def restrict_to_included_studies(stats_df: pl.DataFrame, inclusion: pl.DataFrame) -> pl.DataFrame:
    """Keep primary-study rows whose ``id`` is marked included (drops synthetic rows)."""
    included_ids = inclusion.filter(pl.col("included")).select("id")
    return _primary_study_rows(stats_df).join(included_ids, on="id", how="semi")


def study_effect_row_count(stats_df: pl.DataFrame) -> int:
    return _primary_study_rows(stats_df).height


def effects_with_studies(stats_df: pl.DataFrame) -> list[str]:
    """Distinct effects that have at least one primary-study row, sorted."""
    return sorted(
        _primary_study_rows(stats_df).select("effect").unique().to_series().to_list(),
        key=str,
    )


@dataclass(frozen=True)
class InclusionAudit:
    study_table: pl.DataFrame
    n_included: int
    n_excluded: int
    unrestricted_study_effect_rows: int
    restricted_study_effect_rows: int
    unrestricted_effects: list[str]
    retained_effects: list[str]
    lost_effects: list[str]
    restricted_stats: pl.DataFrame


def inclusion_audit(stats_df: pl.DataFrame, *, max_ts: int = 5) -> InclusionAudit:
    """Build the Core TS≤N inclusion audit for a by-precision statistics frame."""
    primary = _primary_study_rows(stats_df)
    study_table = mark_included_by_ts_count(ts_counts_by_study(primary), max_ts=max_ts)
    restricted = restrict_to_included_studies(primary, study_table)

    unrestricted_effects = effects_with_studies(primary)
    retained_effects = effects_with_studies(restricted)
    lost_effects = [effect for effect in unrestricted_effects if effect not in set(retained_effects)]

    n_included = int(study_table.filter(pl.col("included")).height)
    n_excluded = int(study_table.height - n_included)

    return InclusionAudit(
        study_table=study_table,
        n_included=n_included,
        n_excluded=n_excluded,
        unrestricted_study_effect_rows=study_effect_row_count(primary),
        restricted_study_effect_rows=study_effect_row_count(restricted),
        unrestricted_effects=unrestricted_effects,
        retained_effects=retained_effects,
        lost_effects=lost_effects,
        restricted_stats=restricted,
    )
