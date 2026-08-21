import polars as pl
import pytest
from statsmodels.stats.weightstats import DescrStatsW

from src.data.papers.entities import Papers
from src.experimental_units import (
    collapse_metric_to_units,
    grouping_columns_for_metric,
    unit_columns_for_precision,
    unit_level_statistics,
)


def test_storage_metric_strips_evaluation_context_columns():
    paper = Papers.ALIZADEH.value
    assert grouping_columns_for_metric("storage_size", paper) == ["Model"]
    assert grouping_columns_for_metric("gpu_energy_consumption", paper) == ["Model", "Dataset"]


def test_collapse_metric_to_units_averages_repeated_runs():
    frame = pl.DataFrame(
        {
            "Model": ["m1", "m1", "m2"],
            "Dataset": ["d1", "d1", "d1"],
            "quantization_method": ["ptq", "ptq", "ptq"],
            "precision_configuration": ["w-int8", "w-int8", "w-int8"],
            "inference_latency_improvement": [10.0, 30.0, 5.0],
            "run": [1, 2, 1],
        }
    )
    units = collapse_metric_to_units(
        frame,
        "inference_latency",
        ["Model", "Dataset", "quantization_method", "precision_configuration"],
    )
    assert units.height == 2
    assert units.filter(pl.col("Model") == "m1").select("inference_latency_improvement").item() == 20.0


def test_unit_level_statistics_uses_student_t_mean_interval():
    values = [10.0, 20.0, 30.0]
    stats = unit_level_statistics(pl.Series("value", values))
    lower, upper = DescrStatsW(values).tconfint_mean(alpha=0.05)
    assert stats["n_eff"] == len(values)
    assert stats["mean"] == pytest.approx(sum(values) / len(values))
    assert stats["lower_ci"] == pytest.approx(lower)
    assert stats["upper_ci"] == pytest.approx(upper)


def test_unit_level_statistics_omits_interval_for_n_eff_one_or_zero_variance():
    single = unit_level_statistics(pl.Series("value", [12.0]))
    assert single == {"n_eff": 1, "mean": 12.0, "lower_ci": None, "upper_ci": None}
    identical = [12.0, 12.0, 12.0]
    tied = unit_level_statistics(pl.Series("value", identical))
    assert tied["n_eff"] == len(identical)
    assert tied["mean"] == identical[0]
    assert tied["lower_ci"] is None
    assert tied["upper_ci"] is None


def test_alizadeh_storage_and_energy_precision_unit_counts():
    paper = Papers.ALIZADEH.value
    frame = pl.read_parquet(f"data/processed/{paper.KEY}/improvement_metrics.parquet")

    storage_units = collapse_metric_to_units(
        frame,
        "storage_size",
        unit_columns_for_precision("storage_size", paper),
    ).filter(pl.col("precision_configuration") == "w-int4")
    energy_units = collapse_metric_to_units(
        frame,
        "gpu_energy_consumption",
        unit_columns_for_precision("gpu_energy_consumption", paper),
    ).filter(pl.col("precision_configuration") == "w-int4")

    assert storage_units.height == 18
    assert energy_units.height == 72


def test_sathish_correctness_sample_size_counts_metric_applicable_units_only():
    paper = Papers.SATHISH.value
    frame = pl.read_parquet(f"data/processed/{paper.KEY}/improvement_metrics.parquet")
    subset = frame.filter(pl.col("precision_configuration") == "w-int8, a-int8")

    for metric, expected_n in [("accuracy", 3), ("dsc", 3), ("inference_energy_consumption", 6)]:
        units = collapse_metric_to_units(
            subset,
            metric,
            unit_columns_for_precision(metric, paper),
        )
        assert units.filter(pl.col(f"{metric}_improvement").is_not_null()).height == expected_n


def test_gonzalez_latency_precision_unit_count_ignores_replicates():
    paper = Papers.GONZALEZ.value
    frame = pl.read_parquet(f"data/processed/{paper.KEY}/improvement_metrics.parquet")

    units = collapse_metric_to_units(
        frame,
        "inference_latency",
        unit_columns_for_precision("inference_latency", paper),
    )

    assert frame.height == 28_000
    assert units.height == 28
