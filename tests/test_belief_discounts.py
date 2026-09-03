import math

import polars as pl
import pytest

from src.belief_discounts import (
    discounted_belief,
    sample_size_reliability,
    saturation_parameter,
    second_order_coefficient_of_variation,
    second_order_coefficient_of_variation_expr,
    summarize_effective_sample_sizes,
    variability_reliability,
    variability_reliability_expr,
)


def test_sample_size_reliability_matches_worked_examples():
    assert sample_size_reliability(1, n0=3) == 0.283
    assert sample_size_reliability(2, n0=3) == 0.487
    assert sample_size_reliability(18, n0=3) == 0.998
    assert sample_size_reliability(1) == 0.283
    assert sample_size_reliability(2) == 0.487
    assert sample_size_reliability(18) == 0.998
    assert sample_size_reliability(72) == 1.0


def test_second_order_cv_is_zero_when_std_is_zero():
    assert second_order_coefficient_of_variation(0.0, 75.0) == 0.0
    assert second_order_coefficient_of_variation(0.0, 0.0) == 0.0


def test_second_order_cv_approaches_one_when_mean_near_zero():
    # Independent Kvålseth worked values: σ=29.842, μ=0.485 (S14 accuracy).
    assert second_order_coefficient_of_variation(29.842, 0.485) == pytest.approx(
        29.842 / math.sqrt(29.842**2 + 0.485**2)
    )


def test_variability_reliability_skips_small_n_eff():
    assert variability_reliability(4, std=10.0, mean=5.0) == 1.0
    assert variability_reliability(5, std=0.0, mean=41.3) == 1.0


def test_variability_reliability_matches_alizadeh_energy_v2():
    # σ=8.080, μ=41.3 → V2≈0.192 → e^{-0.1 V2} rounds to 0.981.
    assert variability_reliability(18, std=8.080, mean=41.3) == 0.981


def test_variability_reliability_stable_for_near_zero_mean():
    # σ=29.842, μ=0.485 → V2≈1 → e^{-0.1} rounds to 0.905 (not ~0.044).
    assert variability_reliability(18, std=29.842, mean=0.485) == 0.905


def test_polars_v2_and_alpha_v_match_scalar_helpers():
    frame = pl.DataFrame(
        {
            "n_eff": [4, 18, 18, 18],
            "std": [10.0, 0.0, 8.080, 29.842],
            "mean": [5.0, 75.0, 41.3, 0.485],
        }
    ).with_columns(
        second_order_coefficient_of_variation_expr(pl.col("std"), pl.col("mean")).alias("v2"),
        variability_reliability_expr(pl.col("n_eff"), pl.col("std"), pl.col("mean")).alias("alpha_v"),
    )
    expected_v2 = [
        second_order_coefficient_of_variation(std, mean)
        for std, mean in zip(frame["std"].to_list(), frame["mean"].to_list(), strict=True)
    ]
    expected_alpha_v = [
        variability_reliability(n_eff, std, mean)
        for n_eff, std, mean in zip(
            frame["n_eff"].to_list(), frame["std"].to_list(), frame["mean"].to_list(), strict=True
        )
    ]
    assert frame["v2"].to_list() == pytest.approx(expected_v2)
    assert frame["alpha_v"].to_list() == expected_alpha_v


def test_variability_cutoff_grid_changes_when_alpha_v_applies():
    assert variability_reliability(5, std=8.080, mean=41.3, cutoff=3) == 0.981
    assert variability_reliability(5, std=8.080, mean=41.3, cutoff=8) == 1.0


def test_discounted_belief_uses_both_reliabilities():
    assert discounted_belief(0.713, n_eff=18, std=0.0, mean=75.0, n0=3) == 0.712


def test_saturation_parameter_is_third_quartile_of_n_eff_not_n0():
    assert saturation_parameter([1, 1, 1, 2, 2, 2, 2, 18]) == 2


def test_summarize_effective_sample_sizes_reports_quartiles():
    summary = summarize_effective_sample_sizes([1, 1, 2, 3, 18])
    assert summary.minimum == 1
    assert summary.maximum == 18
    assert summary.n_effects == 5
    assert summary.q3 == pytest.approx(3.0)
