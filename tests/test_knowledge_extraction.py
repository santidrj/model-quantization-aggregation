import polars as pl
import pytest

from src.config import processed_paper_path
from src.data.papers.entities import Papers
from src.data.papers.knowledge_extraction import KnowledgeExtractor
from tests.helpers import assert_frame_equal_with_tolerance, load_json


def normalize_statistics_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col("configuration").cast(pl.String).alias("configuration"))


@pytest.fixture(scope="module")
def dubhir_extractor():
    paper = Papers.DUBHIR.value
    extractor = KnowledgeExtractor(paper.read_data(), paper=paper)
    extractor.extract_knowledge()
    return paper, extractor


def test_improvement_metrics_match_processed_baseline(dubhir_extractor):
    paper, extractor = dubhir_extractor
    expected = pl.read_parquet(processed_paper_path(paper.KEY, "improvement_metrics.parquet"))
    actual = extractor.improvement_metrics
    assert_frame_equal_with_tolerance(
        actual,
        expected,
        sort_by=["library", "Model", "precision_configuration"],
    )


def test_improvement_statistics_match_processed_baseline(dubhir_extractor):
    paper, extractor = dubhir_extractor
    expected_by_configuration = pl.read_parquet(
        processed_paper_path(paper.KEY, "improvement_statistics_by_configuration.parquet")
    )
    expected_by_precision = pl.read_parquet(
        processed_paper_path(paper.KEY, "improvement_statistics_by_precision.parquet")
    )

    actual_by_configuration = normalize_statistics_frame(extractor.get_improvement_statistics())
    actual_by_precision = normalize_statistics_frame(extractor.get_improvement_statistics(by_precision=True))
    expected_by_configuration = normalize_statistics_frame(expected_by_configuration)
    expected_by_precision = normalize_statistics_frame(expected_by_precision)

    assert_frame_equal_with_tolerance(
        actual_by_configuration,
        expected_by_configuration,
        sort_by=["configuration", "effect", "id", "source", "year"],
    )
    assert_frame_equal_with_tolerance(
        actual_by_precision,
        expected_by_precision,
        sort_by=["configuration", "effect", "id", "source", "year"],
    )


def test_effects_match_processed_json_shape_and_keys(dubhir_extractor):
    paper, extractor = dubhir_extractor
    expected_by_configuration = load_json(processed_paper_path(paper.KEY, "effects_by_configuration.json"))
    expected_by_precision = load_json(processed_paper_path(paper.KEY, "effects_by_precision.json"))

    assert len(extractor.effects_by_configuration.to_dicts()) == len(expected_by_configuration)
    assert len(extractor.effects_by_precision.to_dicts()) == len(expected_by_precision)
    assert sorted(extractor.effects_by_configuration.to_dicts()[0].keys()) == sorted(
        expected_by_configuration[0].keys()
    )
    assert sorted(extractor.effects_by_precision.to_dicts()[0].keys()) == sorted(
        expected_by_precision[0].keys()
    )
