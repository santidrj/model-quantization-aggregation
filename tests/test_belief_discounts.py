import pytest

from src.belief_discounts import (
    discounted_belief,
    sample_size_reliability,
    saturation_parameter,
    summarize_effective_sample_sizes,
    variability_reliability,
)


def test_sample_size_reliability_matches_worked_examples():
    assert sample_size_reliability(1, n0=3) == 0.283
    assert sample_size_reliability(2, n0=3) == 0.487
    assert sample_size_reliability(18, n0=3) == 0.998
    assert sample_size_reliability(1) == 0.283
    assert sample_size_reliability(2) == 0.487
    assert sample_size_reliability(18) == 0.998
    assert sample_size_reliability(72) == 1.0


def test_variability_reliability_skips_small_n_eff():
    assert variability_reliability(4, iqr=10.0, mean=5.0) == 1.0
    assert variability_reliability(5, iqr=0.0, mean=41.3) == 1.0


def test_variability_reliability_matches_alizadeh_energy_worked_example():
    assert variability_reliability(18, iqr=8.494, mean=41.3) == 0.98


def test_variability_cutoff_grid_changes_when_alpha_v_applies():
    assert variability_reliability(5, iqr=8.494, mean=41.3, cutoff=3) == 0.98
    assert variability_reliability(5, iqr=8.494, mean=41.3, cutoff=8) == 1.0


def test_discounted_belief_uses_both_reliabilities():
    assert discounted_belief(0.713, n_eff=18, iqr=0.0, mean=75.0, n0=3) == 0.712


def test_saturation_parameter_is_third_quartile_of_n_eff_not_n0():
    assert saturation_parameter([1, 1, 1, 2, 2, 2, 2, 18]) == 2


def test_summarize_effective_sample_sizes_reports_quartiles():
    summary = summarize_effective_sample_sizes([1, 1, 2, 3, 18])
    assert summary.minimum == 1
    assert summary.maximum == 18
    assert summary.n_effects == 5
    assert summary.q3 == pytest.approx(3.0)
