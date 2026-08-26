from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ORDERED_INTENSITY_CODES: tuple[str, ...] = (
    "SN",
    "SN-NE",
    "NE",
    "NE-WN",
    "WN",
    "WN-IF",
    "IF",
    "IF-WP",
    "WP",
    "WP-PO",
    "PO",
    "PO-SP",
    "SP",
)

_POSITIVE_LABEL_TO_CODE = {
    "indifferent": "IF",
    "indifferent - weakly positive": "IF-WP",
    "weakly positive": "WP",
    "weakly positive - positive": "WP-PO",
    "positive": "PO",
    "positive - strongly positive": "PO-SP",
    "strongly positive": "SP",
}
_NEGATIVE_LABEL_TO_CODE = {
    "indifferent": "IF",
    "weakly negative - indifferent": "WN-IF",
    "weakly negative": "WN",
    "negative - weakly negative": "NE-WN",
    "negative": "NE",
    "strongly negative - negative": "SN-NE",
    "strongly negative": "SN",
}


@dataclass(frozen=True)
class SignedIntensityInterval:
    """Closed/open span of a Likert atom or adjacent compound on signed relative improvement (%)."""

    code: str
    lower: float
    upper: float
    lower_inclusive: bool
    upper_inclusive: bool

    def contains(self, value: float) -> bool:
        left = value >= self.lower if self.lower_inclusive else value > self.lower
        right = value <= self.upper if self.upper_inclusive else value < self.upper
        return left and right

    def set_latex(self) -> str:
        return r"\{" + ", ".join(self.code.split("-")) + r"\}"

    def interval_latex(self) -> str:
        lower = r"-\infty" if self.lower == -np.inf else f"{self.lower:g}"
        upper = r"\infty" if self.upper == np.inf else f"{self.upper:g}"
        left = "[" if self.lower_inclusive else "("
        right = "]" if self.upper_inclusive else ")"
        return rf"${left}{lower},{upper}{right}$"


def intensity_label_to_code(label: str) -> str:
    """Map an evidence-model intensity phrase to an ordered Likert code."""
    if label in _POSITIVE_LABEL_TO_CODE:
        return _POSITIVE_LABEL_TO_CODE[label]
    if label in _NEGATIVE_LABEL_TO_CODE:
        return _NEGATIVE_LABEL_TO_CODE[label]
    raise ValueError(f"Unknown effect intensity label: {label!r}")


@dataclass(frozen=True)
class IntensityScale:
    """Immutable RI%→intensity cut-points (avoids mutating the EffectIntensity singleton)."""

    strong_effect: int
    strong_moderate_effect: int
    moderate_effect: int
    weak_moderate_effect: int
    weak_effect: int
    weak_indifferent_effect: int

    def get_intensity(self, improvement_metric: float) -> str:
        sign = "negative" if improvement_metric < 0 else "positive"
        improvement = abs(improvement_metric)
        for threshold, label in (
            (self.weak_indifferent_effect, "indifferent"),
            (
                self.weak_effect,
                f"indifferent - weakly {sign}" if sign == "positive" else f"weakly {sign} - indifferent",
            ),
            (self.weak_moderate_effect, f"weakly {sign}"),
            (
                self.moderate_effect,
                f"weakly {sign} - {sign}" if sign == "positive" else f"{sign} - weakly {sign}",
            ),
            (self.strong_moderate_effect, sign),
            (
                self.strong_effect,
                f"{sign} - strongly {sign}" if sign == "positive" else f"strongly {sign} - {sign}",
            ),
        ):
            if improvement <= threshold:
                return label
        return f"strongly {sign}"


def default_correctness_scale(**overrides: int) -> IntensityScale:
    cuts = {
        "strong_effect": 25,
        "strong_moderate_effect": 20,
        "moderate_effect": 15,
        "weak_moderate_effect": 10,
        "weak_effect": 5,
        "weak_indifferent_effect": 2,
    }
    cuts.update(overrides)
    return IntensityScale(**cuts)


def default_resource_scale(**overrides: int) -> IntensityScale:
    cuts = {
        "strong_effect": 50,
        "strong_moderate_effect": 40,
        "moderate_effect": 30,
        "weak_moderate_effect": 20,
        "weak_effect": 10,
        "weak_indifferent_effect": 2,
    }
    cuts.update(overrides)
    return IntensityScale(**cuts)


def signed_intensity_intervals(scale: EffectIntensity | IntensityScale) -> tuple[SignedIntensityInterval, ...]:
    """Partition of the relative-improvement axis matching ``get_intensity`` cut-points."""
    if isinstance(scale, IntensityScale):
        indifferent = scale.weak_indifferent_effect
        weak = scale.weak_effect
        weak_moderate = scale.weak_moderate_effect
        moderate = scale.moderate_effect
        strong_moderate = scale.strong_moderate_effect
        strong = scale.strong_effect
    else:
        indifferent = scale.WEAK_INDIFFERENT_EFFECT
        weak = scale.WEAK_EFFECT
        weak_moderate = scale.WEAK_MODERATE_EFFECT
        moderate = scale.MODERATE_EFFECT
        strong_moderate = scale.STRONG_MODERATE_EFFECT
        strong = scale.STRONG_EFFECT
    return (
        SignedIntensityInterval("SN", -np.inf, -strong, False, False),
        SignedIntensityInterval("SN-NE", -strong, -strong_moderate, True, False),
        SignedIntensityInterval("NE", -strong_moderate, -moderate, True, False),
        SignedIntensityInterval("NE-WN", -moderate, -weak_moderate, True, False),
        SignedIntensityInterval("WN", -weak_moderate, -weak, True, False),
        SignedIntensityInterval("WN-IF", -weak, -indifferent, True, False),
        SignedIntensityInterval("IF", -indifferent, indifferent, True, True),
        SignedIntensityInterval("IF-WP", indifferent, weak, False, True),
        SignedIntensityInterval("WP", weak, weak_moderate, False, True),
        SignedIntensityInterval("WP-PO", weak_moderate, moderate, False, True),
        SignedIntensityInterval("PO", moderate, strong_moderate, False, True),
        SignedIntensityInterval("PO-SP", strong_moderate, strong, False, True),
        SignedIntensityInterval("SP", strong, np.inf, False, False),
    )


def render_intensity_thresholds_table() -> str:
    """LaTeX tabular of the complete functional-suitability and resource/performance thresholds."""
    correctness = {interval.code: interval for interval in signed_intensity_intervals(CorrectnessIntensity())}
    resource = {interval.code: interval for interval in signed_intensity_intervals(EffectIntensity())}
    lines = [
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        (
            r"\textbf{Intensity} & \textbf{Functional suitability (\%)} & "
            r"\textbf{Resource efficiency / performance (\%)} \\"
        ),
        r"\midrule",
    ]
    for code in ORDERED_INTENSITY_CODES:
        interval = resource[code]
        lines.append(
            " & ".join(
                [
                    interval.set_latex(),
                    correctness[code].interval_latex(),
                    interval.interval_latex(),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


class CorrectnessMetrics:
    @staticmethod
    def metrics() -> list[str]:
        return [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "DSC",
            "mAP",
            "mAP@0.5",
            "mAP@0.5:0.95",
            "mIoU",
            "Perplexity",
            "Word Error Rate",
            "BLEU",
        ]


class ResourceEfficiencyMetrics:
    @staticmethod
    def metrics() -> list[str]:
        return [
            "Storage Size",
            "GPU Utilization",
            "GPU Memory Utilization",
            "GPU Power Draw",
            "GPU Energy Consumption",
            "RAM Usage",
            "RAM Energy Consumption",
            "Inference Power Draw",
            "Inference Energy Consumption",
        ]


class PerformanceMetrics:
    @staticmethod
    def metrics() -> list[str]:
        return [
            "Inference Latency",
        ]


class EffectIntensity:
    _instance = None

    @property
    def STRONG_EFFECT(self) -> int:
        return 50

    @property
    def STRONG_MODERATE_EFFECT(self) -> int:
        return 40

    @property
    def MODERATE_EFFECT(self) -> int:
        return 30

    @property
    def WEAK_MODERATE_EFFECT(self) -> int:
        return 20

    @property
    def WEAK_EFFECT(self) -> int:
        return 10

    @property
    def WEAK_INDIFFERENT_EFFECT(self) -> int:
        return 2

    def __new__(cls, *args, **kwargs):
        if cls.__dict__.get("_instance") is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _threshold_labels(self, sign: str) -> list[tuple[int, str]]:
        return [
            (self.WEAK_INDIFFERENT_EFFECT, "indifferent"),
            (
                self.WEAK_EFFECT,
                f"indifferent - weakly {sign}" if sign == "positive" else f"weakly {sign} - indifferent",
            ),
            (self.WEAK_MODERATE_EFFECT, f"weakly {sign}"),
            (self.MODERATE_EFFECT, f"weakly {sign} - {sign}" if sign == "positive" else f"{sign} - weakly {sign}"),
            (self.STRONG_MODERATE_EFFECT, sign),
            (self.STRONG_EFFECT, f"{sign} - strongly {sign}" if sign == "positive" else f"strongly {sign} - {sign}"),
        ]

    def get_intensity(self, improvement_metric) -> str:
        """
        Get the intensity of the effect based on the improvement metric. The improvement should be expressed in
        percentage.

        Params
        ------
        improvement_metric: float
            The improvement metric expressed in percentage.

        Returns
        -------
        str
            The intensity of the effect.
        """

        sign = "negative" if improvement_metric < 0 else "positive"
        improvement = abs(improvement_metric)

        for threshold, label in self._threshold_labels(sign):
            if improvement <= threshold:
                return label

        return f"strongly {sign}"

    def get_ranges(self) -> dict[str, tuple]:
        """
        Get the ranges of the effect intensity.

        Returns
        -------
        dict
            The ranges of the effect intensity.
        """
        return {
            "SN": (-np.inf, -self.STRONG_EFFECT),
            "SN-NE": (-self.STRONG_EFFECT, -self.STRONG_MODERATE_EFFECT),
            "NE": (-self.STRONG_MODERATE_EFFECT, -self.MODERATE_EFFECT),
            "NE-WN": (-self.MODERATE_EFFECT, -self.WEAK_MODERATE_EFFECT),
            "WN": (-self.WEAK_MODERATE_EFFECT, -self.WEAK_EFFECT),
            "WN-IF": (-self.WEAK_EFFECT, -self.WEAK_INDIFFERENT_EFFECT),
            "IF": (-self.WEAK_INDIFFERENT_EFFECT, self.WEAK_INDIFFERENT_EFFECT),
            "IF-WP": (self.WEAK_INDIFFERENT_EFFECT, self.WEAK_EFFECT),
            "WP": (self.WEAK_EFFECT, self.WEAK_MODERATE_EFFECT),
            "WP-PO": (self.WEAK_MODERATE_EFFECT, self.MODERATE_EFFECT),
            "PO": (self.MODERATE_EFFECT, self.STRONG_MODERATE_EFFECT),
            "PO-SP": (self.STRONG_MODERATE_EFFECT, self.STRONG_EFFECT),
            "SP": (self.STRONG_EFFECT, np.inf),
        }


class EnergyIntensity(EffectIntensity):
    pass


class ResourceUsageIntensity(EffectIntensity):
    pass


class LatencyIntensity(EffectIntensity):
    pass


class CorrectnessIntensity(EffectIntensity):
    @property
    def STRONG_EFFECT(self):
        return 25

    @property
    def STRONG_MODERATE_EFFECT(self):
        return 20

    @property
    def MODERATE_EFFECT(self):
        return 15

    @property
    def WEAK_MODERATE_EFFECT(self):
        return 10

    @property
    def WEAK_EFFECT(self):
        return 5
