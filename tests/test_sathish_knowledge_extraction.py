import polars as pl

from src.data.papers.entities import Papers
from src.data.papers.knowledge_extraction import KnowledgeExtractor


def test_sathish_wint8_discounts_use_task_specific_unit_counts():
    paper = Papers.SATHISH.value
    extractor = KnowledgeExtractor(paper.read_data(), paper=paper)
    extractor.extract_knowledge()

    row = extractor.effects_by_precision.filter(pl.col("precision_configuration") == "w-int8, a-int8").row(
        0, named=True
    )
    accuracy = row["accuracy"]
    dsc = row["dsc"]
    energy = row["inference_energy_consumption"]

    assert accuracy["sample_size_discount"] == 0.777
    assert dsc["sample_size_discount"] == 0.777
    assert accuracy["belief"] == 0.306
    assert dsc["belief"] == 0.306
    assert energy["sample_size_discount"] == 0.950
