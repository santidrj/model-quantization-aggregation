"""Selection manifest from frozen screening artifacts."""

import json

import polars as pl

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from src.data.selection.manifest import (
    build_frozen_selection_artifacts,
    build_selection_manifest,
    selection_summary,
    write_selection_manifest,
)
from src.data.selection.title_key import canonical_title_key


def _mini_inputs(tmp_path):
    papers = pl.DataFrame(
        {
            "Title": [
                "Remaining Kept By LLM",
                "Remaining Dropped By LLM",
                "Calibration Nested Positive",
                "Calibration Nested Negative",
                "Full Text Later Excluded",
                "μLayer Near Miss",
            ],
            "Source": [
                "Scopus",
                "Scopus",
                "Scopus",
                "arXiv",
                "Scopus",
                "arXiv",
            ],
        }
    )
    calibration = pl.DataFrame(
        {
            "Title": [
                "Calibration Nested Positive",
                "Calibration Nested Negative",
                "Orphan Calibration Negative",
                "µLayer Near Miss",  # joins via canonical key to search title below
            ],
            "Included": ["y", "n", "n", "n"],
        }
    )
    frozen_screening_pool = pl.DataFrame(
        {
            "Title": [
                "Remaining Kept By LLM",
                "Calibration Nested Positive",
                "Full Text Later Excluded",
            ]
        }
    )
    final_selection = pl.DataFrame(
        {
            "Title": [
                "Remaining Kept By LLM",
                "Calibration Nested Positive",
                "Full Text Later Excluded",
            ],
            "Included": ["y", "n", "y"],
            "Relevance": [1, 0, 1],
        }
    )
    included_studies = [
        ("paperKept", "Remaining Kept By LLM"),
        ("paperSnow", "Snowball Only Included Study"),
    ]
    return papers, calibration, frozen_screening_pool, final_selection, included_studies


def test_build_selection_manifest_dispositions_calibration_and_pool():
    papers, calibration, frozen_screening_pool, final_selection, included = _mini_inputs(None)
    manifest = build_selection_manifest(
        papers=papers,
        calibration=calibration,
        frozen_screening_pool=frozen_screening_pool,
        final_selection=final_selection,
        included_studies=included,
    )

    by_key = {row["title_key"]: row for row in manifest.to_dicts()}

    kept = by_key[canonical_title_key("Remaining Kept By LLM")]
    assert kept["in_search"] is True
    assert kept["source"] == "Scopus"
    assert kept["llm_prescreen_retained"] is True
    assert kept["in_title_abstract_pool"] is True
    assert kept["final_included"] is True
    assert kept["identification_path"] == "database"

    cal_neg = by_key[canonical_title_key("Calibration Nested Negative")]
    assert cal_neg["calibration_member"] is True
    assert cal_neg["calibration_label"] == "n"
    assert cal_neg["source"] == "arXiv"
    assert cal_neg["in_title_abstract_pool"] is False
    assert cal_neg["llm_prescreen_retained"] is False

    orphan = by_key[canonical_title_key("Orphan Calibration Negative")]
    assert orphan["in_search"] is False
    assert orphan["source"] is None
    assert orphan["calibration_orphan"] is True
    assert orphan["calibration_member"] is True

    snow = by_key[canonical_title_key("Snowball Only Included Study")]
    assert snow["final_included"] is True
    assert snow["identification_path"] == "snowball"
    assert snow["in_search"] is False
    assert snow["source"] is None


def test_selection_summary_counts_match_frozen_decomposition_rules():
    papers, calibration, frozen_screening_pool, final_selection, included = _mini_inputs(None)
    manifest = build_selection_manifest(
        papers=papers,
        calibration=calibration,
        frozen_screening_pool=frozen_screening_pool,
        final_selection=final_selection,
        included_studies=included,
    )
    summary = selection_summary(
        manifest,
        calibration_recall_successes=33,
        calibration_recall_trials=35,
        negative_audit_false_negatives=0,
        negative_audit_trials=100,
    )

    assert summary["n_search"] == 6  # noqa: PLR2004
    assert summary["n_search_by_source"] == {"Scopus": 4, "arXiv": 2}
    assert summary["n_calibration_unique"] == 4  # noqa: PLR2004
    assert summary["n_calibration_nested"] == 3  # noqa: PLR2004
    assert summary["n_calibration_orphans"] == 1  # noqa: PLR2004
    assert summary["n_remaining_search"] == 3  # noqa: PLR2004
    assert summary["n_llm_prescreen_retained"] == 2  # noqa: PLR2004
    assert summary["n_calibration_positive"] == 1  # noqa: PLR2004
    assert summary["n_title_abstract_pool"] == 3  # noqa: PLR2004
    assert summary["n_title_abstract_to_fulltext"] == 2  # noqa: PLR2004
    assert summary["n_fulltext_included_database"] == 1  # noqa: PLR2004
    assert summary["n_snowball_included"] == 1  # noqa: PLR2004
    assert summary["n_final_included"] == 2  # noqa: PLR2004
    assert summary["calibration_recall"]["estimate"] == 33 / 35
    assert summary["negative_audit"]["false_negatives"] == 0


def test_write_selection_manifest_writes_parquet_and_summary_json(tmp_path):
    papers, calibration, frozen_screening_pool, final_selection, included = _mini_inputs(tmp_path)
    manifest = build_selection_manifest(
        papers=papers,
        calibration=calibration,
        frozen_screening_pool=frozen_screening_pool,
        final_selection=final_selection,
        included_studies=included,
    )
    summary = selection_summary(
        manifest,
        calibration_recall_successes=33,
        calibration_recall_trials=35,
        negative_audit_false_negatives=0,
        negative_audit_trials=100,
    )
    out_parquet = tmp_path / "selection-manifest.parquet"
    out_json = tmp_path / "selection-manifest-summary.json"
    write_selection_manifest(manifest, summary, out_parquet, out_json)

    assert pl.read_parquet(out_parquet).height == manifest.height
    loaded = json.loads(out_json.read_text())
    assert loaded["n_final_included"] == 2  # noqa: PLR2004


def test_frozen_selection_manifest_matches_published_decomposition():
    _manifest, summary = build_frozen_selection_artifacts(
        interim_dir=INTERIM_DATA_DIR,
        processed_dir=PROCESSED_DATA_DIR,
    )
    assert summary["n_search"] == 1226  # noqa: PLR2004
    assert summary["n_search_by_source"] == {"Scopus": 1176, "arXiv": 50}
    assert summary["n_calibration_unique"] == 200  # noqa: PLR2004
    assert summary["n_calibration_nested"] == 200  # noqa: PLR2004
    assert summary["n_calibration_orphans"] == 0  # noqa: PLR2004
    assert summary["n_remaining_search"] == 1026  # noqa: PLR2004
    assert summary["n_llm_prescreen_retained"] == 536  # noqa: PLR2004
    assert summary["n_calibration_positive"] == 35  # noqa: PLR2004
    assert summary["n_title_abstract_pool"] == 571  # noqa: PLR2004
    assert summary["n_title_abstract_to_fulltext"] == 84  # noqa: PLR2004
    assert summary["n_fulltext_included_database"] == 16  # noqa: PLR2004
    assert summary["n_snowball_included"] == 5  # noqa: PLR2004
    assert summary["n_final_included"] == 21  # noqa: PLR2004
