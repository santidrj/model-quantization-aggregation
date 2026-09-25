"""Record-level selection manifest from frozen screening artifacts."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path

import polars as pl

from src.data.selection.screening_metrics import jeffreys_interval, proportion
from src.data.selection.title_key import canonical_title_key

FULLTEXT_RELEVANCE = frozenset({1, -1})


def _with_key(frame: pl.DataFrame, title_col: str = "Title") -> pl.DataFrame:
    return frame.with_columns(
        pl.col(title_col).map_elements(canonical_title_key, return_dtype=pl.String).alias("title_key")
    )


def build_selection_manifest(
    *,
    papers: pl.DataFrame,
    calibration: pl.DataFrame,
    frozen_screening_pool: pl.DataFrame,
    final_selection: pl.DataFrame,
    included_studies: Iterable[tuple[str, str]],
) -> pl.DataFrame:
    """Build one disposition row per search hit, calibration orphan, and snowball include.

    Frozen authority for the title-abstract screening pool is ``frozen_screening_pool``:
    LLM pre-screen retention of remaining search hits union human-positive
    calibration records. Human-negative calibration records do not re-enter that pool.
    """
    paper_cols = ["Title"] + (["Source"] if "Source" in papers.columns else [])
    papers_k = _with_key(papers.select(paper_cols)).rename({"Title": "title"})
    if "Source" not in papers_k.columns:
        papers_k = papers_k.with_columns(pl.lit(None).cast(pl.String).alias("Source"))
    papers_k = papers_k.rename({"Source": "source"})
    search_keys = set(papers_k["title_key"].to_list())
    search_title_by_key = dict(zip(papers_k["title_key"], papers_k["title"], strict=True))
    search_source_by_key = dict(zip(papers_k["title_key"], papers_k["source"], strict=True))

    calibration_k = (
        _with_key(calibration.select(["Title", "Included"]))
        .unique(subset=["title_key"], keep="first")
        .rename({"Title": "title", "Included": "calibration_label"})
    )
    cal_by_key = {row["title_key"]: row for row in calibration_k.to_dicts()}
    cal_pos_keys = {k for k, row in cal_by_key.items() if row["calibration_label"] == "y"}
    nested_cal_keys = {k for k in cal_by_key if k in search_keys}
    orphan_cal_keys = set(cal_by_key) - nested_cal_keys

    pool_k = _with_key(frozen_screening_pool.select("Title"))
    pool_keys = set(pool_k["title_key"].to_list())
    llm_remaining_keys = {k for k in pool_keys if k not in cal_pos_keys and k in search_keys}

    final_k = _with_key(final_selection.select(["Title", "Included", "Relevance"])).rename(
        {"Title": "title", "Included": "final_workbook_included", "Relevance": "relevance"}
    )
    final_by_key = {row["title_key"]: row for row in final_k.to_dicts()}

    included_list = list(included_studies)
    included_by_key = {
        canonical_title_key(title): {"paper_key": paper_key, "title": title} for paper_key, title in included_list
    }

    universe_keys = set(search_keys) | set(cal_by_key) | set(included_by_key)

    rows: list[dict] = []
    for title_key in sorted(universe_keys):
        cal = cal_by_key.get(title_key)
        final = final_by_key.get(title_key)
        included = included_by_key.get(title_key)
        in_search = title_key in search_keys
        calibration_member = cal is not None
        calibration_label = cal["calibration_label"] if cal else None
        calibration_orphan = title_key in orphan_cal_keys
        llm_prescreen_retained = title_key in llm_remaining_keys
        calibration_positive = title_key in cal_pos_keys
        in_title_abstract_pool = llm_prescreen_retained or calibration_positive
        relevance = final["relevance"] if final else None
        title_abstract_to_fulltext = relevance in FULLTEXT_RELEVANCE if relevance is not None else False
        final_included = included is not None
        if final_included and in_search:
            identification_path = "database"
        elif final_included:
            identification_path = "snowball"
        else:
            identification_path = None

        title = (
            search_title_by_key.get(title_key)
            or (included["title"] if included else None)
            or (cal["title"] if cal else None)
            or (final["title"] if final else title_key)
        )

        rows.append(
            {
                "title": title,
                "title_key": title_key,
                "paper_key": included["paper_key"] if included else None,
                "source": search_source_by_key.get(title_key),
                "in_search": in_search,
                "calibration_member": calibration_member,
                "calibration_label": calibration_label,
                "calibration_orphan": calibration_orphan,
                "llm_prescreen_retained": llm_prescreen_retained,
                "in_title_abstract_pool": in_title_abstract_pool,
                "relevance": relevance,
                "title_abstract_to_fulltext": title_abstract_to_fulltext,
                "final_included": final_included,
                "identification_path": identification_path,
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "title": pl.String,
            "title_key": pl.String,
            "paper_key": pl.String,
            "source": pl.String,
            "in_search": pl.Boolean,
            "calibration_member": pl.Boolean,
            "calibration_label": pl.String,
            "calibration_orphan": pl.Boolean,
            "llm_prescreen_retained": pl.Boolean,
            "in_title_abstract_pool": pl.Boolean,
            "relevance": pl.Int64,
            "title_abstract_to_fulltext": pl.Boolean,
            "final_included": pl.Boolean,
            "identification_path": pl.String,
        },
    ).sort("title_key")


def selection_summary(
    manifest: pl.DataFrame,
    *,
    calibration_recall_successes: int,
    calibration_recall_trials: int,
    negative_audit_false_negatives: int,
    negative_audit_trials: int,
) -> dict:
    """Aggregate PRISMA-style counts and in-sample screening metrics from a manifest."""
    search = manifest.filter(pl.col("in_search"))
    nested_cal = manifest.filter(pl.col("calibration_member") & pl.col("in_search"))
    remaining_keys = set(search["title_key"]) - set(nested_cal["title_key"])

    recall_est = proportion(calibration_recall_successes, calibration_recall_trials)
    recall_low, recall_high = jeffreys_interval(calibration_recall_successes, calibration_recall_trials)
    audit_est = proportion(
        negative_audit_trials - negative_audit_false_negatives,
        negative_audit_trials,
    )
    # Report FN rate for the audit (0/100) with Jeffreys on FN count
    fn_est = proportion(negative_audit_false_negatives, negative_audit_trials)
    fn_low, fn_high = jeffreys_interval(negative_audit_false_negatives, negative_audit_trials)

    n_search_by_source = {
        row["source"]: row["len"]
        for row in (
            search.filter(pl.col("source").is_not_null())
            .group_by("source")
            .len()
            .sort("source")
            .to_dicts()
        )
    }

    return {
        "n_search": search.height,
        "n_search_by_source": n_search_by_source,
        "n_calibration_unique": manifest.filter(pl.col("calibration_member")).height,
        "n_calibration_nested": nested_cal.height,
        "n_calibration_orphans": manifest.filter(pl.col("calibration_orphan")).height,
        "n_calibration_positive": manifest.filter(pl.col("calibration_label") == "y").height,
        "n_remaining_search": len(remaining_keys),
        "n_llm_prescreen_retained": manifest.filter(pl.col("llm_prescreen_retained")).height,
        "n_title_abstract_pool": manifest.filter(pl.col("in_title_abstract_pool")).height,
        "n_title_abstract_to_fulltext": manifest.filter(pl.col("title_abstract_to_fulltext")).height,
        "n_fulltext_included_database": manifest.filter(pl.col("identification_path") == "database").height,
        "n_snowball_included": manifest.filter(pl.col("identification_path") == "snowball").height,
        "n_final_included": manifest.filter(pl.col("final_included")).height,
        "calibration_recall": {
            "successes": calibration_recall_successes,
            "trials": calibration_recall_trials,
            "estimate": recall_est,
            "jeffreys_low": recall_low,
            "jeffreys_high": recall_high,
            "label": "in-sample calibration performance",
        },
        "negative_audit": {
            "false_negatives": negative_audit_false_negatives,
            "trials": negative_audit_trials,
            "true_negative_rate_estimate": audit_est,
            "false_negative_rate_estimate": fn_est,
            "false_negative_rate_jeffreys_low": fn_low,
            "false_negative_rate_jeffreys_high": fn_high,
            "label": "in-sample audit of predicted negatives",
        },
    }


def write_selection_manifest(
    manifest: pl.DataFrame,
    summary: dict,
    parquet_path: Path,
    summary_json_path: Path,
) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(parquet_path)
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


MANIFEST_PARQUET_NAME = "selection-manifest.parquet"
MANIFEST_SUMMARY_NAME = "selection-manifest-summary.json"

# In-sample screening checks reported in the manuscript (not recomputed from live scores).
CALIBRATION_RECALL_SUCCESSES = 33
CALIBRATION_RECALL_TRIALS = 35
NEGATIVE_AUDIT_FALSE_NEGATIVES = 0
NEGATIVE_AUDIT_TRIALS = 100


def load_included_studies_from_processed(processed_dir: Path) -> list[tuple[str, str]]:
    """Load ``(paper_key, title)`` pairs from processed paper metadata."""
    included: list[tuple[str, str]] = []
    for path in sorted(processed_dir.iterdir()):
        meta_path = path / "metadata.json"
        if not path.is_dir() or not meta_path.exists():
            continue
        payload = json.loads(meta_path.read_text())
        included.append((path.name, payload["title"]))
    return included


def build_frozen_selection_artifacts(
    *,
    interim_dir: Path,
    processed_dir: Path,
) -> tuple[pl.DataFrame, dict]:
    """Build the selection manifest and summary from frozen interim screening files."""
    papers = pl.read_csv(interim_dir / "model-quantization-papers.csv", encoding="utf8")
    calibration = pl.read_excel(interim_dir / "model-quantization-papers-200-sample.xlsx")
    frozen_screening_pool = pl.read_excel(interim_dir / "model-quantization-llm-selected-papers.xlsx")
    final_selection = pl.read_excel(interim_dir / "model-quantization-final-selection.xlsx")
    included_studies = load_included_studies_from_processed(processed_dir)
    manifest = build_selection_manifest(
        papers=papers,
        calibration=calibration,
        frozen_screening_pool=frozen_screening_pool,
        final_selection=final_selection,
        included_studies=included_studies,
    )
    summary = selection_summary(
        manifest,
        calibration_recall_successes=CALIBRATION_RECALL_SUCCESSES,
        calibration_recall_trials=CALIBRATION_RECALL_TRIALS,
        negative_audit_false_negatives=NEGATIVE_AUDIT_FALSE_NEGATIVES,
        negative_audit_trials=NEGATIVE_AUDIT_TRIALS,
    )
    return manifest, summary
