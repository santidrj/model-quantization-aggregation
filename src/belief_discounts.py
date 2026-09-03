"""Sample-size and variability discounts for evidence-model-effect belief."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import polars as pl

DEFAULT_SATURATION_SIZE = 3  # integer nearest 2 / ln 2; α_n(2) is the median of 1-e^{-n/n_0}
DEFAULT_VARIABILITY_K = 0.1
DEFAULT_VARIABILITY_CUTOFF = 4


def sample_size_reliability(n_eff: int, n0: float = DEFAULT_SATURATION_SIZE) -> float:
    """α_n = 1 - exp(-n_eff / n0), rounded to three decimals."""
    return round(1 - math.exp(-n_eff / n0), 3)


def second_order_coefficient_of_variation(std: float, mean: float) -> float:
    """Kvålseth V_2 = σ / sqrt(σ² + μ²) on a real-valued sample."""
    scale = math.hypot(std, mean)
    if scale == 0.0:
        return 0.0
    return std / scale


def variability_reliability(  # noqa: PLR0913
    n_eff: int,
    std: float,
    mean: float,
    *,
    k: float = DEFAULT_VARIABILITY_K,
    cutoff: int = DEFAULT_VARIABILITY_CUTOFF,
) -> float:
    """α_v = exp(-k V_2), or 1 when n_eff is at most the cutoff."""
    if n_eff <= cutoff:
        return 1.0
    return round(math.exp(-k * second_order_coefficient_of_variation(std, mean)), 3)


def discounted_belief(  # noqa: PLR0913
    study_belief: float,
    n_eff: int,
    std: float,
    mean: float,
    *,
    n0: float = DEFAULT_SATURATION_SIZE,
    k: float = DEFAULT_VARIABILITY_K,
    cutoff: int = DEFAULT_VARIABILITY_CUTOFF,
) -> float:
    """B' = B α_n α_v, rounded to three decimals."""
    reliability = sample_size_reliability(n_eff, n0) * variability_reliability(
        n_eff, std, mean, k=k, cutoff=cutoff
    )
    return round(study_belief * reliability, 3)


def saturation_parameter(n_effs: Sequence[int]) -> int:
    """Third quartile of evidence-model-effect n_eff (diagnostic; not n_0)."""
    if not n_effs:
        raise ValueError("saturation_parameter requires at least one n_eff")
    quartile = float(pl.Series(list(n_effs), dtype=pl.Float64).quantile(0.75))
    return int(round(quartile))


@dataclass(frozen=True)
class EffectiveSampleSizeSummary:
    mean: float
    std: float
    minimum: int
    q1: float
    q2: float
    q3: float
    maximum: int
    n_effects: int


def summarize_effective_sample_sizes(n_effs: Sequence[int]) -> EffectiveSampleSizeSummary:
    series = pl.Series(list(n_effs), dtype=pl.Float64)
    return EffectiveSampleSizeSummary(
        mean=float(series.mean()),
        std=float(series.std()),
        minimum=int(series.min()),
        q1=float(series.quantile(0.25)),
        q2=float(series.quantile(0.5)),
        q3=float(series.quantile(0.75)),
        maximum=int(series.max()),
        n_effects=len(n_effs),
    )
