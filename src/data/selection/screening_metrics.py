"""Binomial screening metrics with Jeffreys uncertainty intervals."""

from __future__ import annotations

from scipy.stats import beta


def proportion(successes: int, trials: int) -> float:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials inclusive")
    return successes / trials


def jeffreys_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Central Jeffreys interval for a binomial proportion.

    Uses the Beta(successes + 1/2, trials - successes + 1/2) quantiles.
    When successes is 0, the lower bound is clipped to 0; when successes equals
    trials, the upper bound is clipped to 1.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be between 0 and trials inclusive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    alpha = 1.0 - confidence
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0
    a = successes + 0.5
    b = trials - successes + 0.5
    low = float(beta.ppf(lower_q, a, b))
    high = float(beta.ppf(upper_q, a, b))
    if successes == 0:
        low = 0.0
    if successes == trials:
        high = 1.0
    return low, high
