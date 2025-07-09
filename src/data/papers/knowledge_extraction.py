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
EPSILON = 1e-10


class KnowledgeExtractor:
    PRECISION_COLUMN = "quantization_precision"

    def __init__(  # noqa: PLR0913
        self,
        df: pl.DataFrame | pl.LazyFrame,
        paper: Paper,
    ):
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
        self.compute_improvement()
        self.compute_overall_effect()
        self.compute_effects_by_precision()

    def compute_improvement(self) -> pl.DataFrame:
        baseline_data = self.df.filter(pl.col(self.PRECISION_COLUMN) == self.paper.BASELINE_PRECISION).drop(
            self.PRECISION_COLUMN
        )
        quantization_data = self.df.filter(pl.col(self.PRECISION_COLUMN) != self.paper.BASELINE_PRECISION)

        join_key = (
            self.paper.GROUPING_COLUMNS + self.paper.EXPERIMENT_RUN_KEY
            if self.paper.EXPERIMENT_RUN_KEY is not None
            else self.paper.GROUPING_COLUMNS
        )
        if join_key:
            quantization_data = quantization_data.join(
                baseline_data, on=join_key, how="inner", suffix="_baseline"
            ).with_columns(pl.struct(pl.col(*self.paper.GROUPING_COLUMNS, self.PRECISION_COLUMN)).alias("key"))
        else:
            quantization_data = quantization_data.join(baseline_data, how="cross", suffix="_baseline").with_columns(
                pl.struct(pl.col(self.PRECISION_COLUMN)).alias("key")
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

        # if len(self.correctness_columns) > 1:
        #     self.improvement_metrics = self.improvement_metrics.with_columns(
        #         pl.mean_horizontal([f"{metric}_improvement" for metric, _ in self.correctness_columns]).alias(
        #             "correctness_improvement"
        #         ),
        #     )
        #     self.correctness_columns.append(("correctness", "correctness_improvement"))
        # elif len(self.correctness_columns) == 1:
        #     self.improvement_metrics = self.improvement_metrics.with_columns(
        #         pl.col(f"{self.correctness_columns[0][0]}_improvement").alias("correctness_improvement")
        #     )
        #     self.correctness_columns.append(("correctness", "correctness_improvement"))

        if type(self.improvement_metrics) is pl.LazyFrame:
            self.improvement_metrics = self.improvement_metrics.collect()

        if self.paper.ID == Papers.GONZALEZ.value.ID:
            # Replace -inf with -100 as the max positive value for GPU improvement is 100 and having -inf biases the
            # improvement metric to be negative although there are more cases where improvement is positive.
            # Fill NaN with 0 since it means no improvement
            # Note: This is a workaround for the Gonzalez paper, where the GPU utilization randonly reports 0% at some samples
            self.improvement_metrics = self.improvement_metrics.with_columns(
                pl.col("gpu_utilization_improvement").replace(-np.inf, -100)
            ).fill_nan(0)

        return self.improvement_metrics

    def compute_overall_effect(self) -> pl.DataFrame:
        if not hasattr(self, "improvement_metrics"):
            self.compute_improvement()
        self.overall_effects = self.improvement_metrics.select(
            [
                pl.col("^*_improvement$").mean().cast(pl.Float64),
                pl.col("^*_improvement$").std().cast(pl.Float64).name.suffix("_std"),
                pl.col("^*_improvement$").quantile(Q1).name.suffix("_q1"),
                pl.col("^*_improvement$").quantile(Q3).name.suffix("_q3"),
            ]
        )

        self.overall_effects = self._enrich_data(self.overall_effects)

        return self.overall_effects

    def compute_effects_by_precision(self) -> pl.DataFrame:
        if not hasattr(self, "improvement_metrics"):
            self.compute_improvement()
        self.effects_by_precision = self.improvement_metrics.group_by([self.PRECISION_COLUMN]).agg(
            [
                pl.col("^*_improvement$").mean().cast(pl.Float64),
                pl.col("^*_improvement$").std().cast(pl.Float64).name.suffix("_std"),
                pl.col("^*_improvement$").quantile(Q1).name.suffix("_q1"),
                pl.col("^*_improvement$").quantile(Q3).name.suffix("_q3"),
            ]
        )

        self.effects_by_precision = self._enrich_data(self.effects_by_precision).sort(self.PRECISION_COLUMN)

        return self.effects_by_precision

    def _enrich_data(self, effects_data: pl.DataFrame) -> pl.DataFrame:
        enriched_data = self._add_discount(effects_data)
        enriched_data = self._add_belief(enriched_data)

        enriched_data = enriched_data.with_columns(
            *[
                pl.struct(
                    [
                        pl.col(f"{metric}_improvement").cast(pl.Float64).round(3).alias("improvement"),
                        pl.col(f"{metric}_improvement_std").cast(pl.Float64).round(3).alias("std"),
                        pl.col(f"{metric}_improvement_iqr").cast(pl.Float64).round(3).alias("iqr"),
                        pl.col(f"{metric}_improvement_discount").cast(pl.Float64).round(3).alias("discount"),
                        pl.col(f"{metric}_improvement_belief").cast(pl.Float64).round(3).alias("belief"),
                    ]
                ).alias(f"{metric}")
                for metric, _ in self.correctness_columns + self.resource_efficiency_columns
            ]
        ).drop("^*_improvement.*$")

        return self._add_effect_intensity(enriched_data)

    def get_improvement_statistics(self, by_quantization_precision=False, by_study=False) -> pl.DataFrame:
        """
        Returns the improvement statistics.

        Returns
        -------
        pl.DataFrame
            The improvement statistics.
        """
        df = self.improvement_metrics.select(pl.col("key", "^.+_improvement$"))
        if by_study:
            col_names = pl.Series(name="effect", values=df.drop("key").columns)
            stats = pl.from_pandas(
                sms.describe(df.drop("key"), stats=["nobs", "mean", "ci"], alpha=0.05).T
            ).with_columns(col_names)
            eff_df = stats
        else:
            if by_quantization_precision:
                df = df.with_columns(pl.struct(pl.col("key").struct.field(self.PRECISION_COLUMN)).alias("key"))

            eff_df = []
            for k, group in df.group_by("key"):
                key = k[0]
                metrics = group.drop("key")
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

                if by_quantization_precision:
                    beliefs = pl.concat(
                        [
                            self.effects_by_precision.filter(
                                pl.col(self.PRECISION_COLUMN) == key[self.PRECISION_COLUMN]
                            )
                            .unnest(metric)
                            .select("belief")
                            .unique()
                            .rename({"belief": f"{metric}_improvement"})
                            for metric, _ in self.correctness_columns + self.resource_efficiency_columns
                        ],
                        how="horizontal",
                    ).transpose(include_header=True, header_name="effect", column_names=["belief"])
                    stats = stats.join(beliefs, on="effect", how="inner")

                key_vals = pl.Series(
                    "key", [key] * stats.height, dtype=pl.Struct([pl.Field(k, pl.String) for k in key])
                )
                eff_df.append(stats.with_columns(key_vals))
            eff_df = pl.concat(eff_df, how="vertical_relaxed")

        return eff_df.with_columns(
            pl.lit(self.paper.ID).alias("id"),
            pl.lit(self.paper.AUTHOR).alias("source"),
            pl.lit(self.paper.YEAR).alias("year"),
            pl.col("effect")
            .str.replace_all(r"_improvement", "")
            .str.replace_all(r"_", " ")
            .str.to_titlecase()
            .str.replace_all("Gpu", "GPU")
            .str.replace_all("Dsc", "DSC"),
            # pl.when(pl.col("nobs") == 1).then(pl.col("mean")).otherwise(pl.col("upper_ci")).alias("upper_ci"),
            # pl.when(pl.col("nobs") == 1).then(pl.col("mean")).otherwise(pl.col("lower_ci")).alias("lower_ci"),
        ).filter(pl.col("mean").is_not_null())

    def _add_belief(self, df: pl.DataFrame) -> pl.DataFrame:
        metrics = df.select("^*_improvement$").columns
        for metric in metrics:
            df = df.with_columns((self.paper.BELIEF * (1 - pl.col(f"{metric}_discount"))).alias(f"{metric}_belief"))
        return df

    def _add_discount(self, df: pl.DataFrame) -> pl.DataFrame:
        metrics = df.select("^*_improvement$").columns
        for metric in metrics:
            df = df.with_columns(
                (pl.col(f"{metric}_q3") - pl.col(f"{metric}_q1")).round(3).alias(f"{metric}_iqr")
            ).with_columns(
                (1 - np.e ** (-DISCOUNT_FACTOR * ((pl.col(f"{metric}_iqr")) / pl.col(metric)).abs()))
                .round(3)
                .alias(f"{metric}_discount"),
            )
        return df

    def _add_effect_intensity(self, df: pl.DataFrame) -> pl.DataFrame:
        enriched_df = df.collect() if type(df) is pl.LazyFrame else df

        for metric, _ in self.correctness_columns:
            intensities = [
                CorrectnessIntensity().get_intensity(row[0])
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

    def write_json(self, file: PathLike):
        """
        Write the extracted knowledge to a JSON file.

        Parameters
        ----------
        file : PathLike
            Path to the JSON file.
        """
        final_json = {
            "overall": next(self.overall_effects.iter_rows(named=True)),
        }
        for row in self.effects_by_precision.iter_rows(named=True):
            precision = row.pop(self.PRECISION_COLUMN)
            final_json[precision] = row

        with open(file, "w") as f:
            json.dump(final_json, f, indent=4)
