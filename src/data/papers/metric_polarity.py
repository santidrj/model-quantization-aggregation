MINIMIZED_CORRECTNESS_METRICS = frozenset({"perplexity", "word_error_rate"})


def is_minimized_correctness_metric(metric: str) -> bool:
    return metric in MINIMIZED_CORRECTNESS_METRICS
