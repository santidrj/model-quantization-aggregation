# Contents of the `processed` folder

This folder contains processed data and documentation related to the aggregation and evaluation of model quantization studies. Its structure is as follows:

- **evidence-diagrams-mapping.md**: Markdown file mapping systematic studies (S1–S6) to their corresponding evidence diagrams.
- **metadata-template.json**: Template JSON file for metadata associated with processed studies.
- **model-quantization-final-selection.csv**: CSV file listing the final selection of studies included in the aggregation.
- **systematic-studies-quality-evaluation.md**: General quality evaluation questionnaire template for experimental studies, used as a basis for individual study assessments.

Subfolders for each included study (e.g., `alizadehLanguageModelsSoftware2025/`, `geensEnergyCostModelling2024/`, etc.) contain:

- **systematic-studies-quality-evaluation.md**: Study-specific responses to the quality evaluation questionnaire, including direct quotes and assessments for each criterion.
- **metadata.json**: Metadata for the study, such as the evaluated quantization precisions or the models used.
- **effects.json**: JSON file summarizing the effects measured in the study.
- **improvement_metrics.parquet**: Parquet file containing the relative improvement metrics observed with quantization in the study.
- **improvement_statistics.parquet**: Parquet file containing descriptive statistics (i.e., number of observations, mean, and 95% confidence interval) of the relative improvements reported in **improvement_metrics.parquet**.
- **improvement_statistics_by_precision.parquet**: Parquet files containing descriptive statistics of the relative improvements (i.e., number of observations, mean, 95% confidence interval, and belief) reported in **improvement_metrics.parquet** aggregated by quantization method (i.e., precision + components).

Each subfolder is named after the corresponding study and year, and contains all processed data and documentation relevant to that study.