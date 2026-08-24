import polars as pl

from src.data.papers.entities import Papers
from src.data.papers.knowledge_extraction import KnowledgeExtractor


def test_alizadeh_wint4_energy_sample_size_discount_uses_model_family_n_eff():
    paper = Papers.ALIZADEH.value
    extractor = KnowledgeExtractor(paper.read_data(), paper=paper)
    extractor.extract_knowledge()
    row = extractor.effects_by_precision.filter(pl.col("precision_configuration") == "w-int4").row(0, named=True)
    energy = row["gpu_energy_consumption"]
    storage = row["storage_size"]
    stats = extractor.get_improvement_statistics(by_precision=True)
    energy_n = stats.filter(
        (pl.col("configuration").struct.field("precision_configuration") == "w-int4")
        & (pl.col("effect") == "GPU Energy Consumption")
    )["n_eff"].item()
    storage_n = stats.filter(
        (pl.col("configuration").struct.field("precision_configuration") == "w-int4")
        & (pl.col("effect") == "Storage Size")
    )["n_eff"].item()
    assert energy_n == 18
    assert storage_n == 18
    assert energy["sample_size_discount"] == 1.0
    assert storage["sample_size_discount"] == 1.0
    storage_stats = stats.filter(
        (pl.col("configuration").struct.field("precision_configuration") == "w-int4")
        & (pl.col("effect") == "Storage Size")
    )
    assert storage_stats["lower_ci"].item() is None
    assert storage_stats["upper_ci"].item() is None
