import pytest

from src.data.papers.metric_polarity import is_minimized_correctness_metric


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("perplexity", True),
        ("word_error_rate", True),
        ("accuracy", False),
        ("f1_score", False),
        ("dsc", False),
        ("bleu", False),
    ],
)
def test_is_minimized_correctness_metric(metric, expected):
    assert is_minimized_correctness_metric(metric) is expected
