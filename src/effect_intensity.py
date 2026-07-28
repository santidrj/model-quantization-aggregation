import numpy as np


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
            (self.WEAK_EFFECT, f"indifferent - weakly {sign}" if sign == "positive" else f"weakly {sign} - indifferent"),
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
