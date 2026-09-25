import polars as pl
import pytest

from src.data.papers.entities import Papers
from src.data.papers.precision_nomenclature import parse_precision_label

REPRESENTATIVE_PAPERS = (
    Papers.DUBHIR.value,
    Papers.GONZALEZ.value,
    Papers.CHEN.value,
)


def test_gonzalez_dynamic_quantization_is_post_training():
    assert Papers.GONZALEZ.value.QUANTIZATION_METHOD == "ptq"


def test_chen_tabular_extraction_deduplicates_overlapping_tables():
    assert Papers.CHEN.value.read_data().collect().height == 18


@pytest.mark.parametrize("paper", REPRESENTATIVE_PAPERS, ids=lambda paper: paper.KEY)
def test_representative_papers_load_non_empty_data(paper):
    loaded = paper.read_data()
    frame = loaded.collect() if isinstance(loaded, pl.LazyFrame) else loaded
    assert frame.height > 0
    assert paper.QUANTIZATION_PRECISION_COL in frame.columns
    raw_labels = frame.get_column(paper.QUANTIZATION_PRECISION_COL).unique().to_list()
    canonical_labels = {
        parse_precision_label(label, baseline_precision_configuration=paper.BASELINE_PRECISION)[1]
        for label in raw_labels
    }
    assert paper.BASELINE_PRECISION in canonical_labels

    for metric_name, column_name in paper.CORRECTNESS_COLUMNS.metrics():
        assert column_name in frame.columns, f"{paper.KEY} missing correctness column for {metric_name}"

    for metric_name, column_name in paper.RESOURCE_EFFICIENCY_COLUMNS.metrics():
        assert column_name in frame.columns, f"{paper.KEY} missing resource-efficiency column for {metric_name}"

    for grouping_column in paper.GROUPING_COLUMNS or []:
        assert grouping_column in frame.columns

    for configuration_column in paper.CONFIGURATION_COLUMNS or []:
        assert configuration_column in frame.columns

    for run_key in paper.EXPERIMENT_RUN_KEY or []:
        assert run_key in frame.columns
