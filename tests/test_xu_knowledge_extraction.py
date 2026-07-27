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
    precision_configs = keys["precision_configuration"].to_list()
    assert "w-int8, a-int8" in precision_configs
    assert "mixed-1.8" in precision_configs
    assert "mixed-2" in precision_configs
    assert "w-int2, a-int2" in precision_configs
    assert "mixed" not in precision_configs
    assert "w-int3, a-int3" not in precision_configs

    # ADR 0003: within a method, uniform precedes mixed on equal average bit width.
    qat_rows = (
        stats.filter(pl.col("configuration").struct.field("quantization_method") == "qat")
        .select(pl.col("configuration").struct.field("precision_configuration"))
        .to_series()
        .to_list()
    )
    assert qat_rows.index("w-int2, a-int2") < qat_rows.index("mixed-2")
    assert qat_rows.index("mixed-1.8") < qat_rows.index("w-int2, a-int2")
