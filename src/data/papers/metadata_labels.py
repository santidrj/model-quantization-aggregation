"""Display labels for paper metadata characterization (glossary vocabulary)."""

from __future__ import annotations

NO_ENERGY_MEASUREMENT_LABEL = "no energy measurement"

_EVIDENCE_GRANULARITY_LABELS = {
    "comparative": "tabular summary",
    "comparative (charts)": "chart-only summary",
    "precise": "replication package",
}

_ENERGY_MEASUREMENT_METHODS = frozenset(
    {
        "analytical",
        "hardware-based",
        "software-based",
        "model-based",
    }
)


def label_evidence_granularity(stored: str) -> str:
    """Map a stored ``data_quality`` token to an evidence-granularity glossary label."""
    try:
        return _EVIDENCE_GRANULARITY_LABELS[stored]
    except KeyError as exc:
        raise ValueError(f"Unknown evidence granularity token: {stored!r}") from exc


def label_energy_measurement_method(method: str | None) -> str:
    """Map a stored energy measurement method token to a display label.

    ``None`` means the study did not measure energy consumption.
    """
    if method is None:
        return NO_ENERGY_MEASUREMENT_LABEL
    if method not in _ENERGY_MEASUREMENT_METHODS:
        raise ValueError(f"Unknown energy measurement method: {method!r}")
    return method
