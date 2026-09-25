"""Study ID helpers (consecutive S# labels; numeric order)."""

from __future__ import annotations

import re

import polars as pl

_LEAD_AUTHOR_SPLIT = re.compile(r"\s+et al\.|\s+and\s+", re.IGNORECASE)
_STUDY_ID_PATTERN = re.compile(r"^S(\d+)$")


def lead_author_citation_name(author: str) -> str:
    """Lead-author citation phrase used for Study ID tie-breaking within a year."""
    return _LEAD_AUTHOR_SPLIT.split(author, maxsplit=1)[0].strip()


def _parse_study_id_number(label: str) -> int | None:
    match = _STUDY_ID_PATTERN.fullmatch(str(label))
    return int(match.group(1)) if match else None


def study_id_number(study_id: str) -> int:
    number = _parse_study_id_number(study_id)
    if number is None:
        raise ValueError(f"Invalid Study ID: {study_id!r}")
    return number


def study_id_sort_key(label: str) -> tuple[int, int | str]:
    """Numeric Study ID order; non-Study-ID labels sort after by string."""
    number = _parse_study_id_number(label)
    if number is None:
        return (1, str(label))
    return (0, number)


def study_id_numeric_rank(label: str) -> int:
    """Integer rank for dataframe sorts; non-Study-ID labels rank as -1."""
    number = _parse_study_id_number(label)
    return -1 if number is None else number


def study_id_rank_expr(column: str = "id") -> pl.Expr:
    """Polars expression: numeric Study ID rank for sorting (null if not an S#)."""
    return pl.col(column).str.extract(r"^S(\d+)$").cast(pl.Int64)
