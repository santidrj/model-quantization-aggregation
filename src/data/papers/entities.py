from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import polars as pl

from src.config import EXTERNAL_DATA_DIR


@dataclass
class CorrectnessMetrics:
    accuracy: str | None = None
    precision: str | None = None
    recall: str | None = None
    specificity: str | None = None
    f1_score: str | None = None
    auc: str | None = None
    perplexity: str | None = None
    dsc: str | None = None
    mIoU: str | None = None
    mAP: str | None = None
    mAP_5: str | None = None
    mAP_5_95: str | None = None

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
    inference_energy_consumption: str | None = None
    inference_power_draw: str | None = None
    inference_latency: str | None = None
    gpu_energy_consumption: str | None = None
    gpu_power_draw: str | None = None
    gpu_utilization: str | None = None
    gpu_memory_utilization: str | None = None
    ram_energy_consumption: str | None = None
    ram_usage: str | None = None
    storage_size: str | None = None

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
    GROUPING_COLUMNS: list[str] | None = None
    EXPERIMENT_RUN_KEY: list[str] | None = None

    @abstractmethod
    def read_data(self) -> pl.DataFrame | pl.LazyFrame:
        """
        Reads the data from the paper.

        Returns
        -------
        pl.DataFrame | pl.LazyFrame
            The data read from the paper.
        """
        pass


class HashemiPaper(Paper):
    KEY = "hashemiUnderstandingImpactPrecision2017"
    ID = "S1"
    AUTHOR = "Hashemi et al."
    YEAR = 2017
    QUANTIZATION_PRECISION_COL = "Precision (w, in)"
    BASELINE_PRECISION = "fp32"
    BELIEF = 0.4437487500000001
    RESOURCE_EFFICIENCY_COLUMNS = ResourceEfficiencyMetrics(inference_energy_consumption="Energy (uJ)")
    CORRECTNESS_COLUMNS = CorrectnessMetrics(accuracy="Class. Acc. (%)")
    GROUPING_COLUMNS = ["Model", "Dataset"]

    def read_data(self) -> pl.LazyFrame:
        return pl.scan_csv(
            EXTERNAL_DATA_DIR / self.KEY / "paper-data.csv",
            schema_overrides={
                "Class. Acc. (%)": pl.Float32,
                "Energy (uJ)": pl.Float32,
            },
        )


class DenkingerPaper(Paper):
    KEY = "denkingerImpactMemoryVoltage2020"
    ID = "S2"
    AUTHOR = "Denkinger et al."
    YEAR = 2020
    QUANTIZATION_PRECISION_COL = "numeric_format"
    BASELINE_PRECISION = "fp32"
    BELIEF = 0.44707958333333336
    RESOURCE_EFFICIENCY_COLUMNS = ResourceEfficiencyMetrics(ram_energy_consumption="ram_energy_consumption (uJ)")
    CORRECTNESS_COLUMNS = CorrectnessMetrics(accuracy="accuracy")

    def read_data(self) -> pl.LazyFrame:
        return pl.scan_csv(
            EXTERNAL_DATA_DIR / self.KEY / "paper-data.csv",
            schema_overrides={
                "accuracy": pl.Float32,
                "ram_energy_consumption (uJ)": pl.Float32,
            },
        ).with_columns(
            pl.col("numeric_format")
            .str.replace(r"fxp_8_16", "w-q0.8, a-q0.16")
            .str.replace(r"fxp_4_32", "w-q0.4, a-q0.32")
            .str.replace(r"fxp_4_8", "w-q0.4, a-q0.8")
            .alias("numeric_format"),
        )


class BarnellPaper(Paper):
    KEY = "barnellModelQuantizationSynthetic2021"
    ID = "S3"
    AUTHOR = "Barnell et al."
    YEAR = 2021
    QUANTIZATION_PRECISION_COL = "Precision Format"
    BASELINE_PRECISION = "fp32"
    BELIEF = 0.6679141666666667
    RESOURCE_EFFICIENCY_COLUMNS = ResourceEfficiencyMetrics(
        gpu_power_draw="Power Consumed (W)",
        inference_latency="latency",
        gpu_energy_consumption="uJoules/Frame",
    )
    CORRECTNESS_COLUMNS = CorrectnessMetrics(mAP="mAP")
    GROUPING_COLUMNS = ["Device", "Model"]

    def read_data(self) -> pl.LazyFrame:
        return pl.scan_csv(
            EXTERNAL_DATA_DIR / self.KEY / "paper-data.csv",
            schema_overrides={
                "Power Consumed (W)": pl.Float16,
                "FPS": pl.Float16,
                "uJoules/Frame": pl.Float16,
                "mAP": pl.Float16,
            },
        ).with_columns(
            (1 / pl.col("FPS")).alias("latency"),
        )


class VasquezPaper(Paper):
    KEY = "vasquezActivationDensityBased2021"
    ID = "S4"
    AUTHOR = "Vasquez et al."
    YEAR = 2021
    QUANTIZATION_PRECISION_COL = "Precision Format"
    BASELINE_PRECISION = "fp16"
    BELIEF = 0.3804154166666667
    RESOURCE_EFFICIENCY_COLUMNS = ResourceEfficiencyMetrics(
        inference_energy_consumption="Energy Consumption (uJ)",
    )
    CORRECTNESS_COLUMNS = CorrectnessMetrics()
    GROUPING_COLUMNS = ["Model", "Dataset"]

    def read_data(self) -> pl.LazyFrame:
        return pl.scan_csv(
            EXTERNAL_DATA_DIR / self.KEY / "paper-data.csv",
            schema_overrides={
                "Energy Consumption (uJ)": pl.Float32,
            },
        )


class ZhanPaper(Paper):
    KEY = "zhanFieldProgrammableGate2021"
    ID = "S5"
    AUTHOR = "Zhan et al."
    YEAR = 2021
    QUANTIZATION_PRECISION_COL = "Precision Format"
    BASELINE_PRECISION = "fp32"
    BELIEF = 0.6816654166666667
    RESOURCE_EFFICIENCY_COLUMNS = ResourceEfficiencyMetrics(storage_size="Storage Size (MB)")
    CORRECTNESS_COLUMNS = CorrectnessMetrics(accuracy="Accuracy")
    GROUPING_COLUMNS = ["Dataset", "Model"]

    def read_data(self) -> pl.LazyFrame:
        return pl.scan_csv(
            EXTERNAL_DATA_DIR / self.KEY / "paper-data.csv",
            schema_overrides={
                "Accuracy": pl.Float16,
                "Storage size (MB)": pl.Float16,
            },
        )


class PaulPaper(Paper):
    KEY = "paulEnergyEfficientRespiratoryAnomaly2022"
    ID = "S6"
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

    def read_data(self) -> pl.LazyFrame:
        return pl.scan_csv(EXTERNAL_DATA_DIR / self.KEY / "paper-data.csv")


class SathishPaper(Paper):
    KEY = "sathishVerifiableEnergyEfficient2022"
    ID = "S7"
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
    GROUPING_COLUMNS = ["Model", "Dataset"]

    def read_data(self) -> pl.LazyFrame:
        return pl.scan_csv(EXTERNAL_DATA_DIR / self.KEY / "paper-data.csv").rename(
            {"model": "Model", "dataset": "Dataset"}
        )


class TaoPaper(Paper):
    KEY = "taoExperimentalEnergyConsumption2022"
    ID = "S8"
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

    def read_data(self):
        return (
            pl.read_csv(EXTERNAL_DATA_DIR / self.KEY / "paper-data.csv")
            .with_columns(
                ("w-" + pl.col("Weight Encoding") + ", a-" + pl.col("Activation Encoding")).alias(
                    "quantization_precision"
                )
            )
            .filter(pl.col("Pruning Sparsity") == "0%")
            .drop(["Exp", "Pruning Sparsity", "Weight Encoding", "Activation Encoding"])
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
    ID = "S9"
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


class AlizadehPaper(Paper):
    KEY = "alizadehLanguageModelsSoftware2025"
    ID = "S10"
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
    GROUPING_COLUMNS = ["Model", "Dataset"]

    def read_data(self) -> pl.LazyFrame:
        target_file = EXTERNAL_DATA_DIR / self.KEY / "A100.xlsx"
        model_info = pl.read_excel(target_file, sheet_name="model_info", engine="xlsx2csv")
        code_gen_acc = pl.read_excel(target_file, sheet_name="code_gen_eval", engine="xlsx2csv")
        code_gen_eff = pl.read_excel(target_file, sheet_name="code_gen_energy", engine="xlsx2csv")
        bug_fix_acc = pl.read_excel(target_file, sheet_name="bug_fix_eval", engine="xlsx2csv")
        bug_fix_eff = pl.read_excel(target_file, sheet_name="bug_fix_energy", engine="xlsx2csv")
        test_gen_acc = pl.read_excel(target_file, sheet_name="test_gen_eval", engine="xlsx2csv")
        test_gen_eff = pl.read_excel(target_file, sheet_name="test_gen_energy", engine="xlsx2csv")
        doc_gen_acc = pl.read_excel(target_file, sheet_name="doc_gen_eval", engine="xlsx2csv")
        doc_gen_eff = pl.read_excel(target_file, sheet_name="doc_gen_energy", engine="xlsx2csv")

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
            )
            .with_columns(
                pl.col("model_name").str.replace(r"-(fp.*|q.*)", "").alias("model_name"),
                pl.when(pl.col("quantization_level") == "F16")
                .then(pl.lit("fp16"))
                .when(pl.col("quantization_level") == "Q8_0")
                .then(pl.lit("int8"))
                .otherwise(pl.lit("int4"))
                .alias("quantization_level"),
            )
            .rename({"model_name": "Model", "task": "Dataset"})
        )


class AlshammryPaper(Paper):
    KEY = "alshammryQYOLOv5mQuantizationbasedApproach2025"
    ID = "S11"
    AUTHOR = "Alshammry et al."
    YEAR = 2025
    QUANTIZATION_PRECISION_COL = "Precision Format"
    BASELINE_PRECISION = "fp32"
    BELIEF = 0.6833308333333333
    RESOURCE_EFFICIENCY_COLUMNS = ResourceEfficiencyMetrics(
        storage_size="Model Size (MB)",
        inference_latency="FPS (ms)",
    )
    CORRECTNESS_COLUMNS = CorrectnessMetrics(
        precision="Precision", recall="Recall", mAP_5="mAP@0.5", mAP_5_95="mAP@0.5:0.95"
    )

    def read_data(self) -> pl.LazyFrame:
        return pl.scan_csv(
            EXTERNAL_DATA_DIR / self.KEY / "paper-data.csv",
            schema_overrides={
                "Precision": pl.Float32,
                "Recall": pl.Float32,
                "mAP@0.5": pl.Float32,
                "mAP@0.5:0.95": pl.Float32,
                "Model Size (MB)": pl.Float32,
                "FPS (ms)": pl.Float32,
            },
        )


class DeputterPaper(Paper):
    KEY = "deputterPOQThereParetoOptimal2025"
    ID = "S12"
    AUTHOR = "De Putter et al."
    YEAR = 2025
    QUANTIZATION_PRECISION_COL = "quantization_precision"
    BASELINE_PRECISION = "fp16"
    BELIEF = 0.4541629166666667
    RESOURCE_EFFICIENCY_COLUMNS = ResourceEfficiencyMetrics(
        inference_energy_consumption="Energy (mJ)",
    )
    CORRECTNESS_COLUMNS = CorrectnessMetrics(accuracy="Accuracy", mIoU="mIoU")
    GROUPING_COLUMNS = ["Device", "Dataset", "Model", "Filter Multiplier"]

    def read_data(self) -> pl.LazyFrame:
        return pl.scan_csv(
            EXTERNAL_DATA_DIR / self.KEY / "paper-data.csv",
            schema_overrides={
                "Accuracy": pl.Float32,
                "mIoU": pl.Float32,
                "Energy (mJ)": pl.Float32,
                "Filter Multiplier": pl.Float32,
            },
        ).with_columns(
            pl.when((pl.col("Graph Type") == "Single Precision") & (pl.col("Precision") == "int8"))
            .then(pl.lit("full-int8"))
            .when((pl.col("Graph Type") == "Single Precision") & (pl.col("Precision") == "int4"))
            .then(pl.lit("full-int4"))
            .when((pl.col("Graph Type") == "Single Precision") & (pl.col("Precision") == "int2"))
            .then(pl.lit("full-int2"))
            .when(pl.col("Precision") == "w2a4")
            .then(pl.lit("w-int2, a-int4"))
            .when(pl.col("Precision") == "w2a8")
            .then(pl.lit("w-int2, a-int8"))
            .when(pl.col("Precision") == "w4a8")
            .then(pl.lit("w-int4, a-int8"))
            .otherwise(pl.col("Precision"))
            .alias("quantization_precision"),
        )


class GonzalezPaper(Paper):
    KEY = "gonzalezImpactMLOptimization2024"
    ID = "S13"
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
    GROUPING_COLUMNS = ["Model", "Dataset"]
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
            (pl.col("Optimization").n_unique() > 1).over(["Model", "Dataset"])
        )
        return clean_data

    def compute_metrics(self, dataframe: pl.LazyFrame) -> pl.LazyFrame:
        # Calculate classification accuracy, gpu energy and system energy
        return dataframe.with_columns(
            (pl.sum("Correct Prediction") / pl.len()).over(["Optimization", "Model", "Dataset"]).alias("accuracy"),
            (pl.col("avg_power_draw") * pl.col("Total Time")).alias("gpu_energy"),
            (pl.col("avg_load") * pl.col("Total Time")).alias("sys_energy"),
        )

        # return df.group_by(["Optimization", "Model", "Dataset"]).agg(
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
        ).rename({"Datasets": "Dataset"})

        clean_df = self.clean_data(data)
        return self.compute_metrics(clean_df)


class GuerroujPaper(Paper):
    KEY = "guerroujQuantizedObjectDetection2025"
    ID = "S14"
    AUTHOR = "Guerrouj et al."
    YEAR = 2025
    QUANTIZATION_PRECISION_COL = "Precision Format"
    BASELINE_PRECISION = "fp32"
    BELIEF = 0.6720808333333333
    RESOURCE_EFFICIENCY_COLUMNS = ResourceEfficiencyMetrics(
        storage_size="Storage Size",
        inference_latency="inference_latency",
    )
    CORRECTNESS_COLUMNS = CorrectnessMetrics(mAP="mAP")
    GROUPING_COLUMNS = ["Device", "Model"]

    def read_data(self) -> pl.LazyFrame:
        return pl.scan_csv(
            EXTERNAL_DATA_DIR / self.KEY / "paper-data.csv",
            schema_overrides={
                "mAP": pl.Float32,
                "FPS": pl.Float32,
                "Storage Size": pl.Float32,
            },
        ).with_columns((1 / pl.col("FPS")).alias("inference_latency"))


class KhalilPaper(Paper):
    KEY = "khalilEnergyEfficientDeepLearning2025"
    ID = "S15"
    AUTHOR = "Khalil et al."
    YEAR = 2025
    QUANTIZATION_PRECISION_COL = "Precision Format"
    BASELINE_PRECISION = "fp32"
    BELIEF = 0.6608320833333333
    RESOURCE_EFFICIENCY_COLUMNS = ResourceEfficiencyMetrics(
        inference_latency="Inference Time (ms)",
        ram_usage="RAM (KB)",
        storage_size="Flash (KB)",
        inference_energy_consumption="Energy Consumption (mJ)",
        inference_power_draw="Power Draw (mW)",
    )
    CORRECTNESS_COLUMNS = CorrectnessMetrics()
    GROUPING_COLUMNS = ["Frequency (MHz)", "Model"]

    def read_data(self) -> pl.LazyFrame:
        return pl.scan_csv(
            EXTERNAL_DATA_DIR / self.KEY / "paper-data.csv",
            schema_overrides={
                "Inference Time (ms)": pl.Float32,
                "RAM (KB)": pl.Float32,
                "Flash (KB)": pl.Float32,
                "Energy Consumption (mJ)": pl.Float32,
                "Power Draw (mW)": pl.Float32,
            },
        )


class KoliPaper(Paper):
    KEY = "koliEdgeAIPoweredSystem2025"
    ID = "S16"
    AUTHOR = "Koli et al."
    YEAR = 2025
    QUANTIZATION_PRECISION_COL = "Precision Format"
    BASELINE_PRECISION = "fp32"
    BELIEF = 0.6533320833333334
    RESOURCE_EFFICIENCY_COLUMNS = ResourceEfficiencyMetrics(
        storage_size="Model Size (MB)",
    )
    CORRECTNESS_COLUMNS = CorrectnessMetrics()

    def read_data(self) -> pl.LazyFrame:
        return pl.scan_csv(
            EXTERNAL_DATA_DIR / self.KEY / "paper-data.csv",
            schema_overrides={
                "Model Size (MB)": pl.Float32,
            },
        ).with_columns(
            pl.col("Precision Format").str.replace(r"^w8a32$", "w-int8, a-fp32").alias("Precision Format"),
        )


class KrastevaPaper(Paper):
    KEY = "krastevaImplementingDeepNeural2025"
    ID = "S17"
    AUTHOR = "Krasteva et al."
    YEAR = 2025
    QUANTIZATION_PRECISION_COL = "Precision Format"
    BASELINE_PRECISION = "fp32"
    BELIEF = 0.6762487500000001
    RESOURCE_EFFICIENCY_COLUMNS = ResourceEfficiencyMetrics(
        inference_latency="Execution Time (ms)",
        ram_usage="RAM Usage (B)",
        storage_size="Model Size (B)",
        inference_energy_consumption="Energy Consumption (mJ)",
    )
    CORRECTNESS_COLUMNS = CorrectnessMetrics()

    def read_data(self) -> pl.LazyFrame:
        return pl.scan_csv(
            EXTERNAL_DATA_DIR / self.KEY / "paper-data.csv",
            schema={
                "Model": pl.String,
                "Precision Format": pl.String,
                "Model Size (B)": pl.UInt32,
                "SRAM Usage (B)": pl.UInt32,
                "SDRAM Usage (B)": pl.UInt32,
                "Execution Time (ms)": pl.Float32,
                "Energy Consumption (mJ)": pl.Float32,
            },
        ).with_columns(
            (pl.col("SRAM Usage (B)") + pl.col("SDRAM Usage (B)")).alias("RAM Usage (B)"),
            pl.col("Precision Format").str.replace(r"^w8a8$", "int8").alias("Precision Format"),
        )


class PengPaper(Paper):
    KEY = "pengEfficientExpirationDate2025"
    ID = "S18"
    AUTHOR = "Peng et al."
    YEAR = 2025
    QUANTIZATION_PRECISION_COL = "precision_format"
    BASELINE_PRECISION = "fp32"
    BELIEF = 0.6845808333333333
    RESOURCE_EFFICIENCY_COLUMNS = ResourceEfficiencyMetrics(inference_latency="latency", storage_size="model_size")
    CORRECTNESS_COLUMNS = CorrectnessMetrics()
    GROUPING_COLUMNS = ["compute_unit", "Model", "Dataset"]
    EXPERIMENT_RUN_KEY = ["device"]

    def read_data(self) -> pl.LazyFrame:
        return pl.scan_csv(
            EXTERNAL_DATA_DIR / self.KEY / "paper-data.csv",
            schema_overrides={
                "latency": pl.Float32,
                "model_size": pl.Float32,
            },
        )


class Papers(Enum):
    HASHEMI = HashemiPaper()
    DENKINGER = DenkingerPaper()
    BARNELL = BarnellPaper()
    VASQUEZ = VasquezPaper()
    ZHAN = ZhanPaper()
    PAUL = PaulPaper()
    SATHISH = SathishPaper()
    TAO = TaoPaper()
    GEENS = GeensPaper()
    ALIZADEH = AlizadehPaper()
    ALSHAMMRY = AlshammryPaper()
    DEPUTTER = DeputterPaper()
    GONZALEZ = GonzalezPaper()
    GUERROUJ = GuerroujPaper()
    KHALIL = KhalilPaper()
    KOLI = KoliPaper()
    KRASTEVA = KrastevaPaper()
    PENG = PengPaper()
