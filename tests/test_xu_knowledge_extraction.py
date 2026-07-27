import polars as pl
import pytest

from src.data.papers.entities import Papers
from src.data.papers.knowledge_extraction import KnowledgeExtractor

# Switchboard LSTM-RNN lm_id=1 under w-int1, a-int1 (from Xu paper-data.csv).
SWITCHBOARD_LSTM_RNN_ONE_BIT_PPL = 52.4


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


def test_xu_perplexity_improvement_is_negative_when_perplexity_increases():
    paper = Papers.XU.value
    extractor = KnowledgeExtractor(paper.read_data(), paper=paper)
    extractor.compute_improvement()

    # Switchboard LSTM-RNN lm_id=1 baseline: ppl=40.7; 1-bit uniform: ppl=52.4 (worse).
    row = extractor.improvement_metrics.filter(
        (pl.col("dataset") == "Switchboard")
        & (pl.col("Model") == "LSTM-RNN")
        & (pl.col("precision_configuration") == "w-int1, a-int1")
        & (pl.col("ppl") == SWITCHBOARD_LSTM_RNN_ONE_BIT_PPL)
    )
    assert row.height == 1
    improvement = row["perplexity_improvement"].item()
    assert improvement < 0
    assert improvement == pytest.approx(-28.746, rel=1e-3)


def test_xu_word_error_rate_appears_in_processed_effects():
    paper = Papers.XU.value
    extractor = KnowledgeExtractor(paper.read_data(), paper=paper)
    extractor.extract_knowledge()

    assert "word_error_rate" in extractor.effects_by_precision.columns
    assert "word_error_rate" in extractor.effects_by_configuration.columns
    assert "word_error_rate_improvement" in extractor.improvement_metrics.columns


def test_xu_word_error_rate_improvement_is_negative_when_wer_increases():
    paper = Papers.XU.value
    extractor = KnowledgeExtractor(paper.read_data(), paper=paper)
    extractor.compute_improvement()

    # Switchboard LSTM-RNN lm_id=1 baseline: wer=10.8; 1-bit uniform: wer=12.3 (worse).
    row = extractor.improvement_metrics.filter(
        (pl.col("dataset") == "Switchboard")
        & (pl.col("Model") == "LSTM-RNN")
        & (pl.col("precision_configuration") == "w-int1, a-int1")
        & (pl.col("ppl") == SWITCHBOARD_LSTM_RNN_ONE_BIT_PPL)
    )
    assert row.height == 1
    improvement = row["word_error_rate_improvement"].item()
    assert improvement < 0
    assert improvement == pytest.approx(-13.889, rel=1e-3)
