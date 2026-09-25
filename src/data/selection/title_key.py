"""Canonical title keys for joining study-selection artifacts."""

from __future__ import annotations

import unicodedata


def canonical_title_key(title: str | None) -> str:
    """Normalize a paper title for equality joins across screening artifacts.

    Applies Unicode NFKC, unifies micro-sign with Greek mu, casefolds, collapses
    whitespace, and strips a trailing asterisk used in some bibliographic exports.
    """
    if not title:
        return ""
    text = unicodedata.normalize("NFKC", title)
    text = text.replace("\u00b5", "\u03bc")
    text = text.casefold()
    text = " ".join(text.split())
    return text.rstrip("*").rstrip()
