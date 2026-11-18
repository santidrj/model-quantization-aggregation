"""Candidate counts from the snowballing log (``data/interim/snowballing.ods``).

The spreadsheet's ``Stats`` sheet computes ``COUNTUNIQUE(Publication Id)`` over the
``Backward Snowballing`` and ``Forward Snowballing`` sheets. This module recomputes
those counts from the candidate rows with the standard library, so the pipeline
needs no ODS dependency and the counts cannot drift from the log.
"""

from __future__ import annotations

import html
from pathlib import Path
import re
import zipfile

SNOWBALLING_LOG_NAME = "snowballing.ods"
BACKWARD_SHEET = "Backward Snowballing"
FORWARD_SHEET = "Forward Snowballing"
PUBLICATION_ID_HEADER = "Publication Id"

_MAX_REPEATED_COLUMNS = 64


def _sheet_rows(content: str, sheet: str) -> list[list[str]]:
    match = re.search(rf'<table:table table:name="{re.escape(sheet)}".*?</table:table>', content, re.S)
    if match is None:
        raise ValueError(f"Sheet {sheet!r} not found in snowballing log")
    rows: list[list[str]] = []
    for row in re.findall(r"<table:table-row[^>]*>(.*?)</table:table-row>", match.group(0), re.S):
        cells: list[str] = []
        for attrs, body in re.findall(r"<table:table-cell([^>]*?)(?:/>|>(.*?)</table:table-cell>)", row, re.S):
            repeated = re.search(r'number-columns-repeated="(\d+)"', attrs)
            count = min(int(repeated.group(1)), _MAX_REPEATED_COLUMNS) if repeated else 1
            text = html.unescape(re.sub(r"<[^>]+>", "", body or "")).strip()
            cells.extend([text] * count)
        if any(cells):
            rows.append(cells)
    return rows


def _unique_publication_ids(content: str, sheet: str) -> int:
    header, *body = _sheet_rows(content, sheet)
    column = header.index(PUBLICATION_ID_HEADER)
    return len({row[column] for row in body if len(row) > column and row[column]})


def snowballing_candidate_counts(path: Path) -> dict[str, int]:
    """Return unique backward and forward snowballing candidates in the log."""
    content = zipfile.ZipFile(path).read("content.xml").decode("utf-8")
    return {
        "n_snowball_backward_candidates": _unique_publication_ids(content, BACKWARD_SHEET),
        "n_snowball_forward_candidates": _unique_publication_ids(content, FORWARD_SHEET),
    }
