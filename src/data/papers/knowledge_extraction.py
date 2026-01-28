import json
from os import PathLike

import numpy as np
import polars as pl
from statsmodels.stats import descriptivestats as sms

from src.data.papers.entities import CorrectnessMetrics, Paper, Papers
from src.effect_intensity import (
    CorrectnessIntensity,
    EffectIntensity,
    EnergyIntensity,
    LatencyIntensity,
    ResourceUsageIntensity,
)

CORRECTNESS_METRICS = CorrectnessMetrics()
Q1 = 0.25
Q3 = 0.75
DISCOUNT_FACTOR = 0.1
STABILIZATION_SIZE = 3  # Computed as the median of the number of observations per study
EPSILON = 1e-10

STATS_COLUMNS_ORDER = [
    "configuration",
    "n_subjects",
    "effect",
    "nobs",
    "mean",
    "lower_ci",
    "upper_ci",
    "belief",
]


class KnowledgeExtractor:
    PRECISION_COLUMN = "quantization_precision"

    def __init__(  # noqa: PLR0913
        self,
        df: pl.DataFrame,
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
        self.df = df.drop(
            [
                col
                for col in columns
                if col
                not in [self.paper.QUANTIZATION_PRECISION_COL]
                + [col_name for _, col_name in self.correctness_columns + self.resource_efficiency_columns]
                + (self.paper.GROUPING_COLUMNS or [])
                + (self.paper.EXPERIMENT_RUN_KEY or [])
            ]
        ).rename({self.paper.QUANTIZATION_PRECISION_COL: self.PRECISION_COLUMN})

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
            self.PRECISION_COLUMN
        )
        quantization_data = self.df.filter(pl.col(self.PRECISION_COLUMN) != self.paper.BASELINE_PRECISION)

        if self.paper.GROUPING_COLUMNS is not None:
            join_key = (
                self.paper.GROUPING_COLUMNS + self.paper.EXPERIMENT_RUN_KEY
                if self.paper.EXPERIMENT_RUN_KEY is not None
                else self.paper.GROUPING_COLUMNS
            )
            group_key = [variable for variable in self.paper.GROUPING_COLUMNS if variable != "Model"] + [
                self.PRECISION_COLUMN
            ]
            quantization_data = quantization_data.join(
                baseline_data, on=join_key, how="inner", suffix="_baseline"
            ).with_columns(pl.struct(pl.col(*group_key)).alias("configuration"))
        else:
            quantization_data = quantization_data.join(baseline_data, how="cross", suffix="_baseline").with_columns(
                pl.struct(pl.col(self.PRECISION_COLUMN)).alias("configuration")
            )

        # Compute the relative improvement for each metric
        # Note: We use the baseline value to compute the improvement, so we need to replace 0 with a small value
        # to avoid division by zero resulting in NaN values or infinite values.
        self.improvement_metrics = quantization_data.with_columns(
            *[
                ((pl.col(col) - pl.col(f"{col}_baseline")) / pl.col(f"{col}_baseline") * 100)
                .cast(pl.Float64)
                .alias(f"{metric}_improvement")
                for metric, col in self.correctness_columns
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

        self.effects_by_precision = self.improvement_metrics.group_by(self.PRECISION_COLUMN).agg(
            [
                pl.col("^*_improvement$").mean().cast(pl.Float64),
                pl.col("^*_improvement$").std().cast(pl.Float64).name.suffix("_std"),
                pl.col("^*_improvement$").quantile(Q1).name.suffix("_q1"),
                pl.col("^*_improvement$").quantile(Q3).name.suffix("_q3"),
            ]
        )

        if self.paper.GROUPING_COLUMNS is not None:
            sample_size_by_precision = (
                self.improvement_metrics.with_columns(
                    pl.struct(self.paper.GROUPING_COLUMNS + [self.PRECISION_COLUMN]).alias("samples")
                )
                .unique("samples")
                .group_by(self.PRECISION_COLUMN)
                .agg(pl.col("^*_improvement$").count().name.suffix("_sample_size"))
            )
        else:
            sample_size_by_precision = (
                self.improvement_metrics.unique("configuration")
                .group_by(self.PRECISION_COLUMN)
                .agg(pl.col("^*_improvement$").count().name.suffix("_sample_size"))
            )

        self.effects_by_precision = self.effects_by_precision.join(sample_size_by_precision, on=self.PRECISION_COLUMN)

        self.effects_by_precision = (
            self._enrich_data(self.effects_by_precision).sort(self.PRECISION_COLUMN).drop(pl.col("^*_sample_size$"))
        )

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

        self.effects_by_configuration = self.improvement_metrics.group_by("configuration").agg(
            [
                pl.col("^*_improvement$").mean().cast(pl.Float64),
                pl.col("^*_improvement$").std().cast(pl.Float64).name.suffix("_std"),
                pl.col("^*_improvement$").quantile(Q1).name.suffix("_q1"),
                pl.col("^*_improvement$").quantile(Q3).name.suffix("_q3"),
            ]
        )

        if self.paper.GROUPING_COLUMNS is not None:
            sample_size = (
                self.improvement_metrics.with_columns(
                    pl.struct(self.paper.GROUPING_COLUMNS + [self.PRECISION_COLUMN]).alias("samples")
                )
                .unique("samples")
                .group_by("configuration")
                .agg(pl.col("^*_improvement$").count().name.suffix("_sample_size"))
            )
        else:
            sample_size = (
                self.improvement_metrics.unique("configuration")
                .group_by("configuration")
                .agg(pl.col("^*_improvement$").count().name.suffix("_sample_size"))
            )

        self.effects_by_configuration = self.effects_by_configuration.join(sample_size, on="configuration")

        self.effects_by_configuration = (
            self._enrich_data(self.effects_by_configuration)
            .sort(pl.col("configuration").struct.field(self.PRECISION_COLUMN))
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
        eff_df = []
        for k, group in df.group_by(pl.col("configuration").struct.field(self.PRECISION_COLUMN)):
            key = k[0]
            n_subjects = (
                1
                if self.paper.GROUPING_COLUMNS is None
                else self.improvement_metrics.filter(pl.col("configuration").struct.field(self.PRECISION_COLUMN) == key)
                .select("Model")
                .n_unique()
            )
            metrics = group.drop(pl.col("configuration"))
            if group.height == 1:
                stats = (
                    metrics.transpose(include_header=True, header_name="effect", column_names=["mean"])
                    .with_columns(nobs=1, lower_ci=None, upper_ci=None)
                    .select(["effect", "nobs", "mean", "lower_ci", "upper_ci"])
                )
            else:
                # Select only metrics with more than one unique value to avoid NaN in stats
                # This is important for the statsmodels function to work properly
                metrics_with_change = metrics.select(
                    col.name for col in metrics.select(pl.all().n_unique() > 1) if col.all()
                )

                stats = (
                    pl.from_pandas(sms.describe(metrics_with_change, stats=["nobs", "mean", "ci"], alpha=0.05).T)
                    .with_columns(pl.Series(name="effect", values=metrics_with_change.columns))
                    .select(["effect", "nobs", "mean", "lower_ci", "upper_ci"])
                )

                # Add the metrics with no change to the stats
                metrics_no_change = metrics.select(
                    col.name for col in metrics.select(pl.all().n_unique() == 1) if col.all()
                )
                if metrics_no_change.height > 0:
                    no_change_stats = (
                        metrics_no_change.unique()
                        .transpose(include_header=True, header_name="effect", column_names=["mean"])
                        .with_columns(nobs=metrics_no_change.height, lower_ci=None, upper_ci=None)
                    )

                    # Reorder the columns to match the stats DataFrame
                    no_change_stats = no_change_stats.select(stats.columns)
                    stats = pl.concat([stats, no_change_stats], how="vertical_relaxed")

            beliefs = pl.concat(
                [
                    self.effects_by_precision.filter(pl.col(self.PRECISION_COLUMN) == key)
                    .unnest(metric)
                    .select("belief")
                    .unique()
                    .rename({"belief": f"{metric}_improvement"})
                    for metric, _ in self.correctness_columns + self.resource_efficiency_columns
                ],
                how="horizontal",
            ).transpose(include_header=True, header_name="effect", column_names=["belief"])
            stats = stats.join(beliefs, on="effect", how="inner")

            stats = stats.with_columns(
                pl.Series(
                    "configuration", [key] * stats.height, dtype=pl.Struct([pl.Field(self.PRECISION_COLUMN, pl.String)])
                ),
                pl.lit(n_subjects).alias("n_subjects"),
            )

            # Reorder columns
            stats = stats.select(STATS_COLUMNS_ORDER)

            eff_df.append(stats)

        return pl.concat(eff_df, how="vertical_relaxed")

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
        eff_df = []
        for k, group in df.group_by("configuration"):
            key = k[0]
            n_subjects = (
                1
                if self.paper.GROUPING_COLUMNS is None
                else self.improvement_metrics.filter(pl.col("configuration") == key).select("Model").n_unique()
            )
            metrics = group.drop(pl.col("configuration"))
            if group.height == 1:
                stats = metrics.transpose(
                    include_header=True, header_name="effect", column_names=["mean"]
                ).with_columns(nobs=1, lower_ci=None, upper_ci=None)
            else:
                # Select only metrics with more than one unique value to avoid NaN in stats
                # This is important for the statsmodels function to work properly
                metrics_with_change = metrics.select(
                    col.name for col in metrics.select(pl.all().n_unique() > 1) if col.all()
                )

                stats = pl.from_pandas(
                    sms.describe(metrics_with_change, stats=["nobs", "mean", "ci"], alpha=0.05).T
                ).with_columns(pl.Series(name="effect", values=metrics_with_change.columns))

                # Add the metrics with no change to the stats
                metrics_no_change = metrics.select(
                    col.name for col in metrics.select(pl.all().n_unique() == 1) if col.all()
                )
                if metrics_no_change.height > 0:
                    no_change_stats = (
                        metrics_no_change.unique()
                        .transpose(include_header=True, header_name="effect", column_names=["mean"])
                        .with_columns(nobs=metrics_no_change.height, lower_ci=None, upper_ci=None)
                    )

                    # Reorder the columns to match the stats DataFrame
                    no_change_stats = no_change_stats.select(stats.columns)
                    stats = pl.concat([stats, no_change_stats], how="vertical_relaxed")

            beliefs = pl.concat(
                [
                    self.effects_by_configuration.filter(pl.col("configuration") == key)
                    .unnest(metric)
                    .select("belief")
                    .unique()
                    .rename({"belief": f"{metric}_improvement"})
                    for metric, _ in self.correctness_columns + self.resource_efficiency_columns
                ],
                how="horizontal",
            ).transpose(include_header=True, header_name="effect", column_names=["belief"])
            stats = stats.join(beliefs, on="effect", how="inner")

            stats = stats.with_columns(
                pl.Series(
                    "configuration", [key] * stats.height, dtype=pl.Struct([pl.Field(k, pl.String) for k in key])
                ),
                pl.lit(n_subjects).alias("n_subjects"),
            )

            # Reorder columns
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
        df = self.improvement_metrics.select(pl.col("configuration", "^.+_improvement$"))
        col_names = pl.Series(name="effect", values=df.drop("configuration").columns)
        stats = pl.from_pandas(
            sms.describe(df.drop("configuration"), stats=["nobs", "mean", "ci"], alpha=0.05).T
        ).with_columns(col_names)
        return stats

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
            .str.replace_all("Map", "mAP", literal=True),
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
        Calculate a discount factor based on data variability (IQR).

        High variability relative to the mean effect sizes decreases the discount factor.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame containing quartile information (Q1, Q3).

        Returns
        -------
        pl.DataFrame
            DataFrame with added variability discount columns.
        """
        metrics = df.select("^*_improvement$").columns
        for metric in metrics:
            df = df.with_columns(
                (pl.col(f"{metric}_q3") - pl.col(f"{metric}_q1")).round(3).alias(f"{metric}_iqr")
            ).with_columns(
                pl.when(pl.col(f"{metric}_sample_size") > 4)
                .then((np.e ** (-DISCOUNT_FACTOR * (pl.col(f"{metric}_iqr") / (pl.col(metric) + EPSILON).abs()))))
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
