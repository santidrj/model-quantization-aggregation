"""Sensitivity analysis: TS-count study inclusion and coverage audit."""

import polars as pl
import pytest

from src.sensitivity import (
    SYNTHETIC_STAT_IDS,
    effects_with_studies,
    inclusion_audit,
    mark_included_by_ts_count,
    restrict_to_included_studies,
    study_effect_row_count,
    ts_counts_by_study,
)


def _sample_stats() -> pl.DataFrame:
    """Three studies: S1 has 2 TSs, S2 has 6, S3 has 5; plus a synthetic Aggregated row."""
    rows = [
        {"id": "S1", "evidence_id": "e1", "effect": "Accuracy", "mean": 1.0},
        {"id": "S1", "evidence_id": "e2", "effect": "Accuracy", "mean": 2.0},
        {"id": "S1", "evidence_id": "e1", "effect": "Storage Size", "mean": 10.0},
        {"id": "S2", "evidence_id": "e1", "effect": "Accuracy", "mean": 3.0},
        {"id": "S2", "evidence_id": "e2", "effect": "Accuracy", "mean": 4.0},
        {"id": "S2", "evidence_id": "e3", "effect": "Accuracy", "mean": 5.0},
        {"id": "S2", "evidence_id": "e4", "effect": "BLEU", "mean": 6.0},
        {"id": "S2", "evidence_id": "e5", "effect": "BLEU", "mean": 7.0},
        {"id": "S2", "evidence_id": "e6", "effect": "BLEU", "mean": 8.0},
        {"id": "S3", "evidence_id": "e1", "effect": "Accuracy", "mean": 9.0},
        {"id": "S3", "evidence_id": "e2", "effect": "Accuracy", "mean": 10.0},
        {"id": "S3", "evidence_id": "e3", "effect": "Accuracy", "mean": 11.0},
        {"id": "S3", "evidence_id": "e4", "effect": "Accuracy", "mean": 12.0},
        {"id": "S3", "evidence_id": "e5", "effect": "Accuracy", "mean": 13.0},
        {"id": "Aggregated", "evidence_id": None, "effect": "Accuracy", "mean": 0.0},
    ]
    return pl.DataFrame(rows)


def test_ts_counts_match_unique_evidence_id_per_study():
    counts = ts_counts_by_study(_sample_stats()).sort("id")
    assert counts.to_dicts() == [
        {"id": "S1", "ts_count": 2},
        {"id": "S2", "ts_count": 6},
        {"id": "S3", "ts_count": 5},
    ]


def test_ts_counts_exclude_synthetic_ids():
    counts = ts_counts_by_study(_sample_stats())
    assert set(counts["id"].to_list()).isdisjoint(SYNTHETIC_STAT_IDS)


def test_mark_included_uses_inclusive_max_ts():
    marked = mark_included_by_ts_count(ts_counts_by_study(_sample_stats()), max_ts=5).sort("id")
    by_id = {row["id"]: row["included"] for row in marked.to_dicts()}
    assert by_id == {"S1": True, "S2": False, "S3": True}


def test_restrict_keeps_only_included_primary_studies():
    stats = _sample_stats()
    inclusion = mark_included_by_ts_count(ts_counts_by_study(stats), max_ts=5)
    restricted = restrict_to_included_studies(stats, inclusion)
    assert set(restricted["id"].unique().to_list()) == {"S1", "S3"}


def test_inclusion_audit_reports_lost_effects_and_coverage_delta():
    stats = _sample_stats().filter(pl.col("id") != "Aggregated")
    audit = inclusion_audit(stats, max_ts=5)

    # Fixture: S1+S3 included (2), S2 excluded (1); 14 primary rows → 8 after filter; BLEU only on S2.
    assert audit.n_included == 2  # noqa: PLR2004
    assert audit.n_excluded == 1
    assert audit.unrestricted_study_effect_rows == 14  # noqa: PLR2004
    assert audit.restricted_study_effect_rows == 8  # noqa: PLR2004
    assert audit.lost_effects == ["BLEU"]
    assert "Accuracy" in audit.retained_effects
    assert "Storage Size" in audit.retained_effects

    study_table = audit.study_table.sort("id")
    assert study_table.select("id", "ts_count", "included").to_dicts() == [
        {"id": "S1", "ts_count": 2, "included": True},
        {"id": "S2", "ts_count": 6, "included": False},
        {"id": "S3", "ts_count": 5, "included": True},
    ]


def test_effects_with_studies_ignores_synthetic_rows():
    stats = _sample_stats()
    assert effects_with_studies(stats) == ["Accuracy", "BLEU", "Storage Size"]


def test_study_effect_row_count_ignores_synthetic_rows():
    assert study_effect_row_count(_sample_stats()) == 14  # noqa: PLR2004


def test_max_ts_must_be_positive():
    with pytest.raises(ValueError, match="max_ts"):
        mark_included_by_ts_count(ts_counts_by_study(_sample_stats()), max_ts=0)
