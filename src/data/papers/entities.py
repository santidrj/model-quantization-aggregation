from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import polars as pl

from src.config import EXTERNAL_DATA_DIR


@dataclass
class CorrectnessMetrics:
    accuracy: str = None
    precision: str = None
    recall: str = None
    f1_score: str = None
    auc: str = None
    perplexity: str = None
    dsc: str = None

    def __post_init__(self):
        self._metrics = [(metric, col_name) for metric, col_name in self.__dict__.items() if col_name is not None]

    def metrics(self) -> list[tuple[str, str]]:
        """
        Returns a list of all non-None attributes of the class.

        Returns
        -------
        list
            The list of all non-None attributes of the class.
        """
        return self._metrics


@dataclass
class ResourceEfficiencyMetrics:
    inference_energy_consumption: str = None
    inference_power_draw: str = None
    inference_latency: str = None
    gpu_energy_consumption: str = None
    gpu_power_draw: str = None
    gpu_utilization: str = None
    gpu_memory_utilization: str = None
    storage_size: str = None

    def __post_init__(self):
        self._metrics = [(metric, col_name) for metric, col_name in self.__dict__.items() if col_name is not None]

    def metrics(self) -> list[tuple[str, str]]:
        """
        Returns a list of all non-None attributes of the class."

        Returns
        -------
        list
            The list of all non-None attributes of the class.
        """
        return self._metrics


class Paper(ABC):
    KEY: str
    ID: str
    AUTHOR: str
    YEAR: int
    QUANTIZATION_PRECISION_COL: str
    BASELINE_PRECISION: str
    BELIEF: float
    RESOURCE_EFFICIENCY_COLUMNS: ResourceEfficiencyMetrics
    CORRECTNESS_COLUMNS: CorrectnessMetrics
    GROUPING_COLUMNS: list[str] = None
    EXPERIMENT_RUN_KEY: list[str] = None

    @abstractmethod
    def read_data(self) -> pl.LazyFrame:
        pass


class PaulPaper(Paper):
    KEY = "paulEnergyEfficientRespiratoryAnomaly2022"
    ID = "S1"
    AUTHOR = "Paul et al."
    YEAR = 2022
    QUANTIZATION_PRECISION_COL = "quantization_precision"
    BASELINE_PRECISION = "fp64"
    BELIEF = 0.37333333333333335
    RESOURCE_EFFICIENCY_COLUMNS = ResourceEfficiencyMetrics(
        inference_energy_consumption="inference_energy",
        storage_size="model_size_bits",
    )
    CORRECTNESS_COLUMNS = CorrectnessMetrics(accuracy="accuracy")
    GROUPING_COLUMNS = None

    def read_data(self) -> pl.LazyFrame:
        return pl.scan_csv(EXTERNAL_DATA_DIR / self.KEY / "paper-data.csv")


class SathishPaper(Paper):
    KEY = "sathishVerifiableEnergyEfficient2022"
    ID = "S2"
    AUTHOR = "Sathish et al."
    YEAR = 2022
    QUANTIZATION_PRECISION_COL = "quantization_precision"
    BASELINE_PRECISION = "fp32"
    BELIEF = 0.3939814814814815
    RESOURCE_EFFICIENCY_COLUMNS = ResourceEfficiencyMetrics(
        inference_energy_consumption="system_energy_J",
        storage_size="model_size_MB",
    )
    CORRECTNESS_COLUMNS = CorrectnessMetrics(accuracy="accuracy", dsc="dsc")
    GROUPING_COLUMNS = ["model", "dataset"]

    def read_data(self) -> pl.LazyFrame:
        return pl.scan_csv(EXTERNAL_DATA_DIR / self.KEY / "paper-data.csv")


class TaoPaper(Paper):
    KEY = "taoExperimentalEnergyConsumption2022"
    ID = "S3"
    AUTHOR = "Tao et al."
    YEAR = 2022
    QUANTIZATION_PRECISION_COL = "quantization_precision"
    BASELINE_PRECISION = "w-fp32, a-fp32"
    BELIEF = 0.6429166666666667
    RESOURCE_EFFICIENCY_COLUMNS = ResourceEfficiencyMetrics(
        inference_energy_consumption="system_energy",
        inference_power_draw="system_power",
        inference_latency="inference_latency",
        storage_size="model_size",
    )
    CORRECTNESS_COLUMNS = CorrectnessMetrics(accuracy="accuracy", f1_score="f1_score")
    GROUPING_COLUMNS = None

    def read_data(self) -> pl.LazyFrame:
        return (
            pl.read_csv(EXTERNAL_DATA_DIR / self.KEY / "paper-data.csv")
            .with_columns(
                ("w-" + pl.col("Weight Encoding") + ", a-" + pl.col("Activation Encoding")).alias(
                    "quantization_precision"
                )
            )
            .filter(pl.col("Pruning Sparsity") == "0%")
            .drop(pl.col("Exp", "Pruning Sparsity", "Weight Encoding", "Activation Encoding"))
        ).rename(
            {
                "Accuracy (%)": "accuracy",
                "F1 Score": "f1_score",
                "Model Size (KB)": "model_size",
                "Power Consumption (mW)": "system_power",
                "Inference Time (ms)": "inference_latency",
                "Energy Consumption (µJ)": "system_energy",
            }
        )


class GeensPaper(Paper):
    KEY = "geensEnergyCostModelling2024"
    ID = "S4"
    AUTHOR = "Geens et al."
    YEAR = 2024
    QUANTIZATION_PRECISION_COL = "quantization_precision"
    BASELINE_PRECISION = "w-fp32, a-fp32"
    BELIEF = 0.185
    RESOURCE_EFFICIENCY_COLUMNS = ResourceEfficiencyMetrics(
        inference_energy_consumption="inference_energy",
        inference_latency="inference_clock_cycles",
    )
    CORRECTNESS_COLUMNS = CorrectnessMetrics()
    GROUPING_COLUMNS = None

    def read_data(self) -> pl.LazyFrame:
        baseline_energy = (
            pl.read_csv(EXTERNAL_DATA_DIR / self.KEY / "w32a32-energy.csv").select(pl.col("y") * 1e14).sum()
        )
        baseline_latency = (
            pl.read_csv(EXTERNAL_DATA_DIR / self.KEY / "w32a32-latency.csv").select(pl.col("y") * 1e10).sum()
        )

        w4a16_energy = pl.read_csv(EXTERNAL_DATA_DIR / self.KEY / "w4a16-energy.csv").select(pl.col("y") * 1e14).sum()
        w4a16_latency = pl.read_csv(EXTERNAL_DATA_DIR / self.KEY / "w4a16-latency.csv").select(pl.col("y") * 1e9).sum()

        w1a32_energy = pl.read_csv(EXTERNAL_DATA_DIR / self.KEY / "w1a32-energy.csv").select(pl.col("y") * 1e14).sum()
        w1a32_latency = pl.read_csv(EXTERNAL_DATA_DIR / self.KEY / "w1a32-latency.csv").select(pl.col("y") * 1e9).sum()

        return pl.LazyFrame(
            {
                "quantization_precision": ["w-fp32, a-fp32", "w-int4, a-fp16", "w-int1, a-fp32"],
                "inference_energy": pl.concat([baseline_energy, w4a16_energy, w1a32_energy]),
                "inference_clock_cycles": pl.concat([baseline_latency, w4a16_latency, w1a32_latency]),
            }
        )


class GonzalezPaper(Paper):
    KEY = "gonzalezImpactMLOptimization2024"
    ID = "S5"
    AUTHOR = "Gonzalez Alvarez et al."
    YEAR = 2024
    QUANTIZATION_PRECISION_COL = "Optimization"
    BASELINE_PRECISION = "no_optimization"
    BELIEF = 0.7361083333333334
    RESOURCE_EFFICIENCY_COLUMNS = ResourceEfficiencyMetrics(
        inference_energy_consumption="sys_energy",
        inference_power_draw="avg_load",
        gpu_energy_consumption="gpu_energy",
        gpu_power_draw="avg_power_draw",
        gpu_utilization="avg_utilization_gpu",
        inference_latency="Total Time",
        storage_size="Model Size",
    )
    CORRECTNESS_COLUMNS = CorrectnessMetrics(accuracy="accuracy")
    GROUPING_COLUMNS = ["Model", "Datasets"]
    EXPERIMENT_RUN_KEY = ["Experiment", "Image ID"]

    def clean_data(self, raw_data: pl.LazyFrame) -> pl.LazyFrame:
        # Get only quantization data and baseline
        quantization_data = raw_data.filter(
            pl.col("Optimization").is_in(["no_optimization", "dynamic_quantization"])
        ).with_columns(
            pl.col("Optimization").str.replace("dynamic_quantization", "int8"),
        )

        # Check if there are any missing values
        quantization_data_without_nulls = quantization_data.drop_nulls()

        # Remove models that either have no baseline or no quantization data
        clean_data = quantization_data_without_nulls.filter(
            (pl.col("Optimization").n_unique() > 1).over(["Model", "Datasets"])
        )
        return clean_data

    def compute_metrics(self, dataframe: pl.LazyFrame) -> pl.LazyFrame:
        # Calculate classification accuracy, gpu energy and system energy
        return dataframe.with_columns(
            (pl.sum("Correct Prediction") / pl.len()).over(["Optimization", "Model", "Datasets"]).alias("accuracy"),
            (pl.col("avg_power_draw") * pl.col("Total Time")).alias("gpu_energy"),
            (pl.col("avg_load") * pl.col("Total Time")).alias("sys_energy"),
        )

        # return df.group_by(["Optimization", "Model", "Datasets"]).agg(
        #     avg_accuracy=pl.mean("accuracy"),
        #     model_size=pl.mean("Model Size"),
        #     avg_gpu_power=pl.mean("avg_power_draw"),
        #     avg_gpu_usage=pl.mean("avg_utilization_gpu"),
        #     avg_gpu_energy=pl.mean("gpu_energy"),
        #     avg_inference_energy=pl.mean("sys_energy"),
        #     avg_inference_power=pl.mean("avg_load"),
        #     avg_inference_latency=pl.mean("Total Time"),
        # )

    def read_data(self) -> pl.LazyFrame:
        data = pl.scan_csv(
            EXTERNAL_DATA_DIR / self.KEY / "final_ds_image-classification.csv",
            has_header=True,
            separator=",",
            schema_overrides={
                "Optimization": str,
                "Model": str,
                "y_pred": str,
                "y_true": str,
                "Correct Prediction": pl.UInt8,
                "Total Time": pl.Float64,
                "Model Size": pl.UInt32,  # Model size is in bytes
                "avg_utilization_gpu": pl.Float64,
                "avg_power_draw": pl.Float64,
                "avg_load": pl.Float64,
            },
        )

        data = data.select(
            [
                "Experiment",
                "Optimization",
                "Model",
                "Datasets",
                "Image ID",
                "Correct Prediction",
                "Total Time",
                "Model Size",
                "avg_utilization_gpu",
                "avg_power_draw",
                "avg_load",
            ]
        )

        clean_df = self.clean_data(data)
        return self.compute_metrics(clean_df)


class AlizadehPaper(Paper):
    KEY = "alizadehLanguageModelsSoftware2025"
    ID = "S6"
    AUTHOR = "Alizadeh et al."
    YEAR = 2025
    QUANTIZATION_PRECISION_COL = "quantization_level"
    BASELINE_PRECISION = "fp16"
    BELIEF = 0.7129601851851852
    RESOURCE_EFFICIENCY_COLUMNS = ResourceEfficiencyMetrics(
        gpu_energy_consumption="total_energy_J",
        gpu_power_draw="mean_gpu_power",
        gpu_utilization="mean_gpu_util",
        gpu_memory_utilization="mean_gpu_mem_util",
        storage_size="estimated_size_MB",
        inference_latency="total_elapsed_time",
    )
    CORRECTNESS_COLUMNS = CorrectnessMetrics(accuracy="accuracy")
    GROUPING_COLUMNS = ["model_name", "task"]

    def read_data(self) -> pl.LazyFrame:
        target_file = EXTERNAL_DATA_DIR / self.KEY / "A100.xlsx"
        model_info = pl.read_excel(target_file, sheet_name="model_info")
        code_gen_acc = pl.read_excel(target_file, sheet_name="code_gen_eval")
        code_gen_eff = pl.read_excel(target_file, sheet_name="code_gen_energy")
        bug_fix_acc = pl.read_excel(target_file, sheet_name="bug_fix_eval")
        bug_fix_eff = pl.read_excel(target_file, sheet_name="bug_fix_energy")
        test_gen_acc = pl.read_excel(target_file, sheet_name="test_gen_eval")
        test_gen_eff = pl.read_excel(target_file, sheet_name="test_gen_energy")
        doc_gen_acc = pl.read_excel(target_file, sheet_name="doc_gen_eval")
        doc_gen_eff = pl.read_excel(target_file, sheet_name="doc_gen_energy")

        # Merge accuracy with energy
        merged_code_gen = code_gen_acc.join(code_gen_eff, on="model_name")
        merged_code_gen_full = (
            merged_code_gen.join(model_info, on="model_name")
            .with_columns(pl.lit("code_gen").alias("task"))
            .rename({"pass@1": "accuracy"})
        )
        merged_bug_fix = bug_fix_acc.join(bug_fix_eff, on="model_name")
        merged_bug_fix_full = (
            merged_bug_fix.join(model_info, on="model_name")
            .with_columns(pl.lit("bug_fix").alias("task"))
            .rename({"pass@1": "accuracy"})
        )
        merged_test_gen = test_gen_acc.join(test_gen_eff, on="model_name")
        merged_test_gen_full = (
            merged_test_gen.join(model_info, on="model_name")
            .with_columns(pl.lit("test_gen").alias("task"))
            .rename({"correctness": "accuracy"})
        )
        merged_doc_gen = doc_gen_acc.join(doc_gen_eff, on="model_name")
        merged_doc_gen_full = (
            merged_doc_gen.join(model_info, on="model_name")
            .with_columns(pl.lit("doc_gen").alias("task"))
            .rename({"pass@1": "accuracy"})
        )

        return pl.LazyFrame(
            pl.concat(
                [
                    merged_code_gen_full,
                    merged_bug_fix_full,
                    merged_test_gen_full,
                    merged_doc_gen_full,
                ],
                how="diagonal",
            ).with_columns(
                pl.col("model_name").str.replace(r"-(fp.*|q.*)", "").alias("model_name"),
                pl.when(pl.col("quantization_level") == "F16")
                .then(pl.lit("fp16"))
                .when(pl.col("quantization_level") == "Q8_0")
                .then(pl.lit("int8"))
                .otherwise(pl.lit("int4"))
                .alias("quantization_level"),
            )
        )


class Papers(Enum):
    PAUL = PaulPaper()
    SATHISH = SathishPaper()
    TAO = TaoPaper()
    GEENS = GeensPaper()
    GONZALEZ = GonzalezPaper()
    ALIZADEH = AlizadehPaper()
