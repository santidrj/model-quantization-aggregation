import polars as pl

from src.data.papers.entities import Papers
from src.data.papers.knowledge_extraction import KnowledgeExtractor


def test_xu_configuration_struct_uses_method_and_precision_configuration():
    paper = Papers.XU.value
    extractor = KnowledgeExtractor(paper.read_data(), paper=paper)
    extractor.extract_knowledge()

    configuration_fields = extractor.improvement_metrics["configuration"].dtype.fields
    assert [field.name for field in configuration_fields] == [
        "dataset",
        "quantization_method",
        "precision_configuration",
        "quantization_configuration",
    ]


def test_xu_by_precision_statistics_split_qat_and_ptq():
    paper = Papers.XU.value
    extractor = KnowledgeExtractor(paper.read_data(), paper=paper)
    extractor.extract_knowledge()

    stats = extractor.get_improvement_statistics(by_precision=True)
    keys = (
        stats.select(
            pl.col("configuration").struct.field("quantization_method"),
            pl.col("configuration").struct.field("precision_configuration"),
        )
        .unique()
        .sort(["quantization_method", "precision_configuration"])
    )
    assert "qat" in keys["quantization_method"].to_list()
    assert "ptq" in keys["quantization_method"].to_list()
    assert "w-int8, a-int8" in keys["precision_configuration"].to_list()
