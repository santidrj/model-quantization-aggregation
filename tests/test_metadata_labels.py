import pytest

from src.data.papers.metadata_labels import (
    NO_ENERGY_MEASUREMENT_LABEL,
    label_energy_measurement_method,
    label_evidence_granularity,
)


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ("comparative", "tabular summary"),
        ("comparative (charts)", "chart-only summary"),
        ("precise", "replication package"),
    ],
)
def test_label_evidence_granularity(stored, expected):
    assert label_evidence_granularity(stored) == expected


def test_label_evidence_granularity_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown evidence granularity"):
        label_evidence_granularity("summary statistics")


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (None, NO_ENERGY_MEASUREMENT_LABEL),
        ("analytical", "analytical"),
        ("hardware-based", "hardware-based"),
        ("software-based", "software-based"),
        ("model-based", "model-based"),
    ],
)
def test_label_energy_measurement_method(stored, expected):
    assert label_energy_measurement_method(stored) == expected


def test_label_energy_measurement_method_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown energy measurement method"):
        label_energy_measurement_method("wattmeter")
