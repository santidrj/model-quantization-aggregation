import polars as pl

from src.data.papers.entities import Papers
from src.data.papers.knowledge_extraction import KnowledgeExtractor


def test_xu_configuration_struct_uses_bit_width_and_full_configuration():
    paper = Papers.XU.value
    extractor = KnowledgeExtractor(paper.read_data(), paper=paper)
    extractor.extract_knowledge()

    configuration_fields = extractor.improvement_metrics["configuration"].dtype.fields
    assert [field.name for field in configuration_fields] == [
        "dataset",
        "quantization_precision",
        "quantization_configuration",
    ]

    precisions = (
        extractor.improvement_metrics.select(pl.col("configuration").struct.field("quantization_precision"))
        .unique()
        .to_series()
        .to_list()
    )
    assert set(precisions) == {"int1", "int2", "int3", "int4", "int8"}

    switchboard_int4 = extractor.improvement_metrics.filter(
        (pl.col("configuration").struct.field("dataset") == "Switchboard")
        & (pl.col("configuration").struct.field("quantization_precision") == "int4")
    )
    assert switchboard_int4.select(pl.col("configuration").struct.field("quantization_configuration")).n_unique() > 1


def test_xu_by_precision_statistics_use_bit_width_labels():
    paper = Papers.XU.value
    extractor = KnowledgeExtractor(paper.read_data(), paper=paper)
    extractor.extract_knowledge()

    stats = extractor.get_improvement_statistics(by_precision=True)
    precision_values = stats.select(pl.col("configuration").struct.field("quantization_precision")).unique().to_series()
    assert set(precision_values.to_list()) == {"int1", "int2", "int3", "int4", "int8"}
