import json
from os import PathLike

import numpy as np
import polars as pl

from src.belief_discounts import (
    DEFAULT_SATURATION_SIZE,
    DEFAULT_VARIABILITY_CUTOFF,
    DEFAULT_VARIABILITY_K,
)
from src.data.papers.entities import CorrectnessMetrics, Paper, Papers
from src.data.papers.metric_polarity import is_minimized_correctness_metric
from src.data.papers.precision_nomenclature import (
    normalize_quantization_method,
    parse_precision_label,
    precision_configuration_sort_key,
)
from src.effect_intensity import (
    CorrectnessIntensity,
    EffectIntensity,
    EnergyIntensity,
    LatencyIntensity,
    ResourceUsageIntensity,
)
from src.experimental_units import (
    cluster_columns_from_unit_columns,
    collapse_metric_to_units,
    unit_columns_for_configuration,
    unit_columns_for_precision,
    unit_level_statistics,
)

CORRECTNESS_METRICS = CorrectnessMetrics()
Q1 = 0.25
Q3 = 0.75
DISCOUNT_FACTOR = DEFAULT_VARIABILITY_K
STABILIZATION_SIZE = DEFAULT_SATURATION_SIZE
MIN_SAMPLE_SIZE_FOR_VARIABILITY_DISCOUNT = DEFAULT_VARIABILITY_CUTOFF

STATS_COLUMNS_ORDER = [
    "configuration",
    "effect",
    "n_eff",
    "mean",
    "lower_ci",
    "upper_ci",
    "belief",
]


def _cluster_sample_size_expr(unit_columns: list[str], improvement_column: str) -> pl.Expr:
    cluster_columns = cluster_columns_from_unit_columns(unit_columns)
    return (
        pl.struct(cluster_columns)
        .filter(pl.col(improvement_column).is_not_null())
        .n_unique()
        .alias(f"{improvement_column}_sample_size")
    )


def _cluster_ids_for_units(units: pl.DataFrame, unit_columns: list[str]) -> pl.Series:
    cluster_columns = cluster_columns_from_unit_columns(unit_columns)
    cluster_key = pl.concat_str([pl.col(column).cast(pl.String) for column in cluster_columns], separator="\0")
    return units.select(cluster_key).to_series()


class KnowledgeExtractor:
    PRECISION_COLUMN = "precision_configuration"
    METHOD_COLUMN = "quantization_method"

    def __init__(  # noqa: PLR0913
        self,
        df: pl.DataFrame | pl.LazyFrame,
        paper: Paper,
    ):
        """
        Initialize the KnowledgeExtractor.

        Parameters
        ----------
        df : pl.DataFrame
            The input DataFrame containing the raw experimental data.
        paper : Paper
            The metadata object representing the paper being analyzed.
        """
        self.paper = paper

        self.correctness_columns = paper.CORRECTNESS_COLUMNS.metrics()
        self.resource_efficiency_columns = paper.RESOURCE_EFFICIENCY_COLUMNS.metrics()

        columns = df.collect_schema().names() if type(df) is pl.LazyFrame else df.columns
        method_source = self.paper.QUANTIZATION_METHOD_COL
        keep = (
            [self.paper.QUANTIZATION_PRECISION_COL]
            + ([method_source] if method_source and method_source in columns else [])
            + [col_name for _, col_name in self.correctness_columns + self.resource_efficiency_columns]
            + (self.paper.GROUPING_COLUMNS or [])
            + (self.paper.CONFIGURATION_COLUMNS or [])
            + (self.paper.EXPERIMENT_RUN_KEY or [])
        )
        self.df = df.drop([col for col in columns if col not in keep]).rename(
            {self.paper.QUANTIZATION_PRECISION_COL: self.PRECISION_COLUMN}
        )
        if method_source and method_source in keep and method_source != self.METHOD_COLUMN:
            self.df = self.df.rename({method_source: self.METHOD_COLUMN})

        self.df = self._canonicalize_precision_and_method(self.df)

    def _canonicalize_precision_and_method(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        columns = df.collect_schema().names() if type(df) is pl.LazyFrame else df.columns
        default_method = self.paper.QUANTIZATION_METHOD
        baseline = self.paper.BASELINE_PRECISION

        def _parse_label(label: str) -> dict[str, str | None]:
            method, config = parse_precision_label(label, baseline_precision_configuration=baseline)
            return {"method": method, "config": config}

        df = (
            df.with_columns(
                pl.col(self.PRECISION_COLUMN)
                .map_elements(_parse_label, return_dtype=pl.Struct({"method": pl.String, "config": pl.String}))
                .alias("_parsed")
            )
            .with_columns(
                pl.col("_parsed").struct.field("config").alias(self.PRECISION_COLUMN),
                pl.col("_parsed").struct.field("method").alias("_method_from_label"),
            )
            .drop("_parsed")
        )

        if self.METHOD_COLUMN in columns:
            df = df.with_columns(
                pl.col(self.METHOD_COLUMN)
                .map_elements(
                    lambda value: normalize_quantization_method(value) if value is not None else None,
                    return_dtype=pl.String,
                )
                .alias(self.METHOD_COLUMN)
            )
            df = df.with_columns(
                pl.coalesce(pl.col(self.METHOD_COLUMN), pl.col("_method_from_label")).alias(self.METHOD_COLUMN)
            )
        else:
            df = df.with_columns(pl.col("_method_from_label").alias(self.METHOD_COLUMN))

        if default_method is not None:
            df = df.with_columns(pl.col(self.METHOD_COLUMN).fill_null(default_method))

        quantized = df.filter(pl.col(self.PRECISION_COLUMN) != baseline)
        null_methods = quantized.select(pl.col(self.METHOD_COLUMN).is_null().any())
        has_null = null_methods.collect().item() if type(null_methods) is pl.LazyFrame else null_methods.item()
        if has_null:
            raise ValueError(
                f"Paper {self.paper.KEY} is missing quantization_method on one or more quantized rows "
                "and has no single QUANTIZATION_METHOD default"
            )

        # Baseline rows are excluded from by-precision aggregation; fill only for a typed column.
        return df.with_columns(pl.col(self.METHOD_COLUMN).fill_null(default_method or "qat")).drop("_method_from_label")

    def _by_precision_key(self) -> list[str]:
        return [self.METHOD_COLUMN, self.PRECISION_COLUMN]

    def _sort_effects_by_precision(self, effects: pl.DataFrame) -> pl.DataFrame:
        """Order by quantization method, then precision configuration sort key (ADR 0003)."""
        methods = effects[self.METHOD_COLUMN].to_list()
        precisions = effects[self.PRECISION_COLUMN].to_list()
        order = sorted(
            range(len(methods)),
            key=lambda i: (methods[i], precision_configuration_sort_key(precisions[i])),
        )
        return effects[order]

    def _sort_frame_by_precision_configuration(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Order rows whose `configuration` struct carries method + precision (ADR 0003)."""
        methods = frame["configuration"].struct.field(self.METHOD_COLUMN).to_list()
        precisions = frame["configuration"].struct.field(self.PRECISION_COLUMN).to_list()
        order = sorted(
            range(len(methods)),
            key=lambda i: (methods[i], precision_configuration_sort_key(precisions[i]), i),
        )
        return frame[order]

    def _configuration_key(self) -> list[str]:
        if self.paper.GROUPING_COLUMNS is None:
            return self._by_precision_key()
        return [
            *[variable for variable in self.paper.GROUPING_COLUMNS if variable != "Model"],
            *self._by_precision_key(),
            *(self.paper.CONFIGURATION_COLUMNS or []),
        ]

    def extract_knowledge(self):
        """
        Extract knowledge by computing improvements and analyzing effects.

        This method orchestrates the calculation of:
        1. Improvement metrics relative to baseline.
        2. Effects aggregated by configuration.
        3. Effects aggregated by precision.
        """
        self.compute_improvement()
        self.compute_effects_by_configuration()
        self.compute_effects_by_precision()

    def compute_improvement(self) -> pl.DataFrame:
        """
        Compute the relative improvement of metrics compared to the baseline.

        The baseline is defined where the precision column matches the paper's baseline precision.
        Calculates percentage improvement for correctness and resource efficiency metrics.

        Returns
        -------
        pl.DataFrame
            DataFrame containing improvement metrics.
        """
        baseline_data = self.df.filter(pl.col(self.PRECISION_COLUMN) == self.paper.BASELINE_PRECISION).drop(
            *self._by_precision_key()
        )
        quantization_data = self.df.filter(pl.col(self.PRECISION_COLUMN) != self.paper.BASELINE_PRECISION)

        if self.paper.GROUPING_COLUMNS is not None:
            join_key = (
                self.paper.GROUPING_COLUMNS + self.paper.EXPERIMENT_RUN_KEY
                if self.paper.EXPERIMENT_RUN_KEY is not None
                else self.paper.GROUPING_COLUMNS
            )
            group_key = self._configuration_key()
            quantization_data = quantization_data.join(
                baseline_data, on=join_key, how="inner", suffix="_baseline"
            ).with_columns(pl.struct(pl.col(*group_key)).alias("configuration"))
        else:
            quantization_data = quantization_data.join(baseline_data, how="cross", suffix="_baseline").with_columns(
                pl.struct(pl.col(*self._by_precision_key())).alias("configuration")
            )

        # Compute the relative improvement for each metric.
        # Polarity (maximized vs minimized) is defined globally; see ADR 0004.
        maximized_correctness_columns = [
            (metric, col) for metric, col in self.correctness_columns if not is_minimized_correctness_metric(metric)
        ]
        minimized_correctness_columns = [
            (metric, col) for metric, col in self.correctness_columns if is_minimized_correctness_metric(metric)
        ]
        self.improvement_metrics = quantization_data.with_columns(
            *[
                ((pl.col(col) - pl.col(f"{col}_baseline")) / pl.col(f"{col}_baseline") * 100)
                .cast(pl.Float64)
                .alias(f"{metric}_improvement")
                for metric, col in maximized_correctness_columns
            ]
            + [
                ((pl.col(f"{col}_baseline") - pl.col(col)) / pl.col(f"{col}_baseline") * 100)
                .cast(pl.Float64)
                .alias(f"{metric}_improvement")
                for metric, col in minimized_correctness_columns
            ]
            + [
                ((pl.col(f"{col}_baseline") - pl.col(col)) / pl.col(f"{col}_baseline") * 100)
                .cast(pl.Float64)
                .alias(f"{metric}_improvement")
                for metric, col in self.resource_efficiency_columns
            ]
        )

        if type(self.improvement_metrics) is pl.LazyFrame:
            self.improvement_metrics = self.improvement_metrics.collect()

        if self.paper.ID == Papers.GONZALEZ.value.ID:
            # Replace -inf with -100 as the max positive value for GPU improvement is 100 and having -inf biases the
            # improvement metric to be negative although there are more cases where improvement is positive.
            # Fill NaN with 0 since it means no improvement
            # Note: This is a workaround for the Gonzalez paper, where the GPU utilization randonly reports 0% at some
            # samples
            self.improvement_metrics = self.improvement_metrics.with_columns(
                pl.col("gpu_utilization_improvement").replace(-np.inf, -100)
            ).fill_nan(0)

        return self.improvement_metrics

    def _metric_names(self) -> list[str]:
        return [metric for metric, _ in self.correctness_columns + self.resource_efficiency_columns]

    def _aggregate_metric_at_precision(self, metric: str) -> pl.DataFrame:
        improvement_column = f"{metric}_improvement"
        unit_columns = unit_columns_for_precision(metric, self.paper)
        units = collapse_metric_to_units(self.improvement_metrics, metric, unit_columns)
        return units.group_by(self._by_precision_key()).agg(
            pl.col(improvement_column).mean().cast(pl.Float64).alias(improvement_column),
            pl.col(improvement_column).std().cast(pl.Float64).alias(f"{improvement_column}_std"),
            pl.col(improvement_column).quantile(Q1).alias(f"{improvement_column}_q1"),
            pl.col(improvement_column).quantile(Q3).alias(f"{improvement_column}_q3"),
            _cluster_sample_size_expr(unit_columns, improvement_column),
        )

    def _aggregate_metric_at_configuration(self, metric: str) -> pl.DataFrame:
        improvement_column = f"{metric}_improvement"
        available_columns = set(self.improvement_metrics.columns)
        unit_columns = unit_columns_for_configuration(metric, self.paper, available_columns)
        units = collapse_metric_to_units(self.improvement_metrics, metric, unit_columns)
        return units.group_by("configuration").agg(
            pl.col(improvement_column).mean().cast(pl.Float64).alias(improvement_column),
            pl.col(improvement_column).std().cast(pl.Float64).alias(f"{improvement_column}_std"),
            pl.col(improvement_column).quantile(Q1).alias(f"{improvement_column}_q1"),
            pl.col(improvement_column).quantile(Q3).alias(f"{improvement_column}_q3"),
            _cluster_sample_size_expr(unit_columns, improvement_column),
        )

    def _join_metric_precision_frames(
        self, frames: list[pl.DataFrame], *, group_columns: list[str] | None = None
    ) -> pl.DataFrame:
        join_on = group_columns or self._by_precision_key()
        combined = frames[0]
        for frame in frames[1:]:
            combined = combined.join(frame, on=join_on, how="full", coalesce=True)
        return combined

    def compute_effects_by_precision(self) -> pl.DataFrame:
        """
        Compute aggregated effects grouped by quantization precision.

        Aggregates improvement metrics to calculate mean, standard deviation, and quartiles.
        Enriches the data with statistical discounting factors.

        Returns
        -------
        pl.DataFrame
            Aggregated effects by precision.
        """
        if not hasattr(self, "improvement_metrics"):
            self.compute_improvement()

        metric_frames = [self._aggregate_metric_at_precision(metric) for metric in self._metric_names()]
        self.effects_by_precision = self._join_metric_precision_frames(metric_frames)
        self.effects_by_precision = self._enrich_data(self.effects_by_precision).drop(pl.col("^*_sample_size$"))
        self.effects_by_precision = self._sort_effects_by_precision(self.effects_by_precision)

        return self.effects_by_precision

    def compute_effects_by_configuration(self) -> pl.DataFrame:
        """
        Compute aggregated effects grouped by experimental configuration.

        Configurations are determined by grouping columns (e.g., model architecture) and precision.
        Aggregates improvement metrics to calculate mean, standard deviation, and quartiles.
        Enriches the data with statistical discounting factors.

        Returns
        -------
        pl.DataFrame
            Aggregated effects by configuration.
        """
        if not hasattr(self, "improvement_metrics"):
            self.compute_improvement()

        metric_frames = [self._aggregate_metric_at_configuration(metric) for metric in self._metric_names()]
        self.effects_by_configuration = self._join_metric_precision_frames(
            metric_frames, group_columns=["configuration"]
        )

        self.effects_by_configuration = (
            self._enrich_data(self.effects_by_configuration)
            .sort(*[pl.col("configuration").struct.field(column) for column in self._configuration_key()])
            .drop(pl.col("^*_sample_size$"))
        )

        return self.effects_by_configuration

    def _enrich_data(self, effects_data: pl.DataFrame) -> pl.DataFrame:
        """
        Enrich effects data with statistical metrics and intensity.

        Adds sample size discounts, variability discounts, belief scores, and classifies
        the intensity of the effects.

        Parameters
        ----------
        effects_data : pl.DataFrame
            The dataframe containing aggregated effects.

        Returns
        -------
        pl.DataFrame
            The enriched DataFrame with additional statistical and intensity columns.
        """
        enriched_data = self._add_sample_size_discount(effects_data)
        enriched_data = self._add_variability_discount(enriched_data)
        enriched_data = self._add_belief(enriched_data)

        enriched_data = enriched_data.with_columns(
            *[
                pl.struct(
                    [
                        pl.col(f"{metric}_improvement").cast(pl.Float64).round(3).alias("improvement"),
                        pl.col(f"{metric}_improvement_std").cast(pl.Float64).round(3).alias("std"),
                        pl.col(f"{metric}_improvement_iqr").cast(pl.Float64).round(3).alias("iqr"),
                        pl.col(f"{metric}_improvement_sample_size_discount")
                        .cast(pl.Float64)
                        .round(3)
                        .alias("sample_size_discount"),
                        pl.col(f"{metric}_improvement_variability_discount")
                        .cast(pl.Float64)
                        .round(3)
                        .alias("variability_discount"),
                        (
                            pl.col(f"{metric}_improvement_sample_size_discount")
                            * pl.col(f"{metric}_improvement_variability_discount")
                        )
                        .round(3)
                        .alias("discount_factor"),
                        (
                            1
                            - (
                                pl.col(f"{metric}_improvement_sample_size_discount")
                                * pl.col(f"{metric}_improvement_variability_discount")
                            )
                        )
                        .round(3)
                        .alias("p_value"),
                        pl.col(f"{metric}_improvement_belief").cast(pl.Float64).round(3).alias("belief"),
                    ]
                ).alias(f"{metric}")
                for metric, _ in self.correctness_columns + self.resource_efficiency_columns
            ]
        ).drop("^*_improvement.*$")

        return self._add_effect_intensity(enriched_data)

    def _statistics_for_metric_group(
        self,
        metric: str,
        subset: pl.DataFrame,
        unit_columns: list[str],
    ) -> dict[str, float | int | str | None]:
        improvement_column = f"{metric}_improvement"
        units = collapse_metric_to_units(subset, metric, unit_columns)
        stats = unit_level_statistics(
            units[improvement_column],
            cluster_ids=_cluster_ids_for_units(units, unit_columns),
        )
        return {"effect": improvement_column, **stats}

    def _get_improvement_statistics_by_precision(self) -> pl.DataFrame:
        """
        Calculate improvement statistics grouped by precision using statsmodels.

        Computes count, mean, and confidence intervals for each metric within each precision group.

        Returns
        -------
        pl.DataFrame
            DataFrame containing detailed statistics for each precision group.
        """
        df = self.improvement_metrics.select(pl.col("configuration", "^.+_improvement$"))
        precision_key = pl.struct(
            pl.col("configuration").struct.field(self.METHOD_COLUMN),
            pl.col("configuration").struct.field(self.PRECISION_COLUMN),
        )
        eff_df = []
        for k, _group in df.group_by(precision_key):
            key = k[0]
            method = key[self.METHOD_COLUMN]
            precision = key[self.PRECISION_COLUMN]
            subset = self.improvement_metrics.filter(
                (pl.col(self.METHOD_COLUMN) == method) & (pl.col(self.PRECISION_COLUMN) == precision)
            )
            stats = pl.DataFrame(
                [
                    self._statistics_for_metric_group(
                        metric,
                        subset,
                        unit_columns_for_precision(metric, self.paper),
                    )
                    for metric in self._metric_names()
                ]
            )
            beliefs = self._collect_beliefs(
                self.effects_by_precision.filter(
                    (pl.col(self.METHOD_COLUMN) == method) & (pl.col(self.PRECISION_COLUMN) == precision)
                ),
            )
            stats = stats.join(beliefs, on="effect", how="inner")

            stats = stats.with_columns(
                pl.Series(
                    "configuration",
                    [key] * stats.height,
                    dtype=pl.Struct(
                        [
                            pl.Field(self.METHOD_COLUMN, pl.String),
                            pl.Field(self.PRECISION_COLUMN, pl.String),
                        ]
                    ),
                )
            )
            stats = stats.select(STATS_COLUMNS_ORDER)
            eff_df.append(stats)

        return self._sort_frame_by_precision_configuration(pl.concat(eff_df, how="vertical_relaxed"))

    def _get_improvement_statistics_by_configuration(self) -> pl.DataFrame:
        """
        Calculate improvement statistics grouped by configuration using statsmodels.

        Computes count, mean, and confidence intervals for each metric within each configuration.

        Returns
        -------
        pl.DataFrame
            DataFrame containing detailed statistics for each configuration.
        """
        df = self.improvement_metrics.select(pl.col("configuration", "^.+_improvement$"))
        available_columns = set(self.improvement_metrics.columns)
        eff_df = []
        for k, _group in df.group_by("configuration"):
            key = k[0]
            subset = self.improvement_metrics.filter(pl.col("configuration") == key)
            stats = pl.DataFrame(
                [
                    self._statistics_for_metric_group(
                        metric,
                        subset,
                        unit_columns_for_configuration(metric, self.paper, available_columns),
                    )
                    for metric in self._metric_names()
                ]
            )
            beliefs = self._collect_beliefs(
                self.effects_by_configuration.filter(pl.col("configuration") == key),
            )
            stats = stats.join(beliefs, on="effect", how="inner")

            stats = stats.with_columns(
                pl.Series(
                    "configuration", [key] * stats.height, dtype=pl.Struct([pl.Field(k, pl.String) for k in key])
                ),
            )
            stats = stats.select(STATS_COLUMNS_ORDER)
            eff_df.append(stats)

        return pl.concat(eff_df, how="vertical_relaxed")

    def _get_improvement_statistics_by_study(self) -> pl.DataFrame:
        """
        Calculate overall improvement statistics for the entire study.

        Computes count, mean, and confidence intervals for each metric across the entire dataset.

        Returns
        -------
        pl.DataFrame
            DataFrame containing detailed statistics for the study.
        """
        stats = pl.DataFrame(
            [
                self._statistics_for_metric_group(
                    metric,
                    self.improvement_metrics,
                    unit_columns_for_precision(metric, self.paper),
                )
                for metric in self._metric_names()
            ]
        )
        return stats

    def _collect_beliefs(self, effects_frame: pl.DataFrame) -> pl.DataFrame:
        return pl.concat(
            [
                effects_frame.unnest(metric).select("belief").unique().rename({"belief": f"{metric}_improvement"})
                for metric, _ in self.correctness_columns + self.resource_efficiency_columns
            ],
            how="horizontal",
        ).transpose(include_header=True, header_name="effect", column_names=["belief"])

    def get_improvement_statistics(self, by_precision=False, by_study=False) -> pl.DataFrame:
        """
        Returns the improvement statistics. By default, the statistics are grouped by experimental configuration.

        Parameters
        ----------
        by_precision : bool, optional
            If True, the statistics are grouped by precision, by default False.

        by_study : bool, optional
            If True, the statistics are grouped by study, by default False.

        Returns
        -------
        pl.DataFrame
            The improvement statistics.
        """
        if by_precision:
            eff_df = self._get_improvement_statistics_by_precision()
        elif by_study:
            eff_df = self._get_improvement_statistics_by_study()
        else:
            eff_df = self._get_improvement_statistics_by_configuration()

        return eff_df.with_columns(
            pl.lit(self.paper.ID).alias("id"),
            pl.lit(self.paper.AUTHOR).alias("source"),
            pl.lit(self.paper.YEAR).alias("year"),
            pl.col("effect")
            .str.replace_all(r"_improvement", "")
            .str.replace_all("_", " ")
            .str.to_titlecase()
            .str.replace_all("Gpu", "GPU")
            .str.replace_all("Ram", "RAM")
            .str.replace_all("Dsc", "DSC")
            .str.replace_all("Miou", "mIoU")
            .str.replace_all("Map 5 95", "mAP@0.5:0.95", literal=True)
            .str.replace_all("Map 5", "mAP@0.5", literal=True)
            .str.replace_all("Map", "mAP", literal=True)
            .str.replace_all("Bleu", "BLEU", literal=True),
            # pl.when(pl.col("nobs") == 1).then(pl.col("mean")).otherwise(pl.col("upper_ci")).alias("upper_ci"),
            # pl.when(pl.col("nobs") == 1).then(pl.col("mean")).otherwise(pl.col("lower_ci")).alias("lower_ci"),
        ).filter(pl.col("mean").is_not_null())

    def _add_belief(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate belief scores for improvement metrics.

        Belief is derived from the paper's base belief score modulated by sample size
        and variability discounts.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame containing improvement metrics and discounts.

        Returns
        -------
        pl.DataFrame
            DataFrame with added belief columns.
        """
        metrics = df.select("^*_improvement$").columns
        for metric in metrics:
            df = df.with_columns(
                (
                    self.paper.BELIEF
                    * (pl.col(f"{metric}_sample_size_discount") * pl.col(f"{metric}_variability_discount"))
                ).alias(f"{metric}_belief")
            )
        return df

    def _add_sample_size_discount(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate a discount factor based on sample size.

        Larger sample sizes result in a smaller discount (closer to 1), penalizing
        small sample sizes.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame containing sample size information.

        Returns
        -------
        pl.DataFrame
            DataFrame with added sample size discount columns.
        """
        metrics = df.select("^*_improvement$").columns
        for metric in metrics:
            df = df.with_columns(
                (1 - np.e ** -(pl.col(f"{metric}_sample_size") / STABILIZATION_SIZE))
                .round(3)
                .alias(f"{metric}_sample_size_discount"),
            )

        return df

    def _add_variability_discount(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate a discount factor from Kvålseth's second-order CV (V_2).

        α_v = exp(-k V_2) with V_2 = σ / sqrt(σ² + μ²) on experimental-unit relative
        improvements; α_v = 1 when cluster n_eff is at most the cutoff.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame containing mean, std, and quartile information.

        Returns
        -------
        pl.DataFrame
            DataFrame with added variability discount columns.
        """
        metrics = df.select("^*_improvement$").columns
        for metric in metrics:
            scale_sq = pl.col(f"{metric}_std") ** 2 + pl.col(metric) ** 2
            v2 = (
                pl.when(scale_sq == 0)
                .then(pl.lit(0.0))
                .otherwise(pl.col(f"{metric}_std") / scale_sq.sqrt())
            )
            df = df.with_columns(
                (pl.col(f"{metric}_q3") - pl.col(f"{metric}_q1")).round(3).alias(f"{metric}_iqr")
            ).with_columns(
                pl.when(pl.col(f"{metric}_sample_size") > MIN_SAMPLE_SIZE_FOR_VARIABILITY_DISCOUNT)
                .then(np.e ** (-DISCOUNT_FACTOR * v2))
                .otherwise(pl.lit(1))
                .round(3)
                .alias(f"{metric}_variability_discount")
            )
        return df

    def _add_effect_intensity(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Classify the intensity/magnitude of the effects.

        Uses domain-specific intensity classifiers (e.g. EnergyIntensity) to label
        the magnitude of improvements.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame with improvement metrics.

        Returns
        -------
        pl.DataFrame
            DataFrame with added intensity classification columns.
        """
        enriched_df = df.collect() if type(df) is pl.LazyFrame else df

        for metric, _ in self.correctness_columns:
            intensities = [
                CorrectnessIntensity().get_intensity(row[0]) if row[0] is not None else None
                for row in enriched_df.select(pl.col(metric).struct.field("improvement")).iter_rows()
            ]
            enriched_df = enriched_df.with_columns(
                pl.col(metric).struct.with_fields(
                    intensity=pl.Series(intensities),
                )
            )

        for metric, _ in self.resource_efficiency_columns:
            if "energy" in metric:
                intensities = [
                    EnergyIntensity().get_intensity(row[0])
                    for row in enriched_df.select(pl.col(metric).struct.field("improvement")).iter_rows()
                ]
            elif "utilization" in metric:
                intensities = [
                    ResourceUsageIntensity().get_intensity(row[0])
                    for row in enriched_df.select(pl.col(metric).struct.field("improvement")).iter_rows()
                ]
            elif "latency" in metric:
                intensities = [
                    LatencyIntensity().get_intensity(row[0])
                    for row in enriched_df.select(pl.col(metric).struct.field("improvement")).iter_rows()
                ]
            else:
                intensities = [
                    EffectIntensity().get_intensity(row[0])
                    for row in enriched_df.select(pl.col(metric).struct.field("improvement")).iter_rows()
                ]
            enriched_df = enriched_df.with_columns(pl.col(metric).struct.with_fields(intensity=pl.Series(intensities)))

        return enriched_df

    def save_effects_by_configuration(self, file: PathLike):
        """
        Write the extracted effects grouped by configuration to a JSON file.

        Parameters
        ----------
        file: PathLike
            Path to the JSON file.
        """
        with open(file, "w") as f:
            json.dump(self.effects_by_configuration.to_dicts(), f, indent=4)

    def save_effects_by_precision(self, file: PathLike):
        """
        Write the extracted effects grouped by target quantization precision to a JSON file.

        Parameters
        ----------
        file: PathLike
            Path to the JSON file.
        """
        with open(file, "w") as f:
            json.dump(self.effects_by_precision.to_dicts(), f, indent=4)
