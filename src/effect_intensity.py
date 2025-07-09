from typing import Dict, List

import numpy as np


class EffectIntensity:
    @property
    @staticmethod
    def STRONG_EFFECT(self) -> int:
        return 50

    @property
    @staticmethod
    def STRONG_MODERATE_EFFECT(self) -> int:
        return 40

    @property
    @staticmethod
    def MODERATE_EFFECT(self) -> int:
        return 30

    @property
    @staticmethod
    def WEAK_MODERATE_EFFECT(self) -> int:
        return 20

    @property
    @staticmethod
    def WEAK_EFFECT(self) -> int:
        return 10

    @property
    @staticmethod
    def WEAK_INDIFERENT_EFFECT(self) -> int:
        return 2

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EffectIntensity, cls).__new__(cls)
        return cls._instance

    def get_intensity(self, improvement_metric) -> str:
        """
        Get the intensity of the effect based on the improvement metric. The improvement should be exressed in
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

        if improvement <= self.WEAK_INDIFERENT_EFFECT:
            return "indiferent"
        elif improvement > self.WEAK_INDIFERENT_EFFECT and improvement <= self.WEAK_EFFECT:
            return f"indiferent - weakly {sign}" if sign == "positive" else f"weakly {sign} - indiferent"
        elif improvement > self.WEAK_EFFECT and improvement <= self.WEAK_MODERATE_EFFECT:
            return f"weakly {sign}"
        elif improvement > self.WEAK_MODERATE_EFFECT and improvement <= self.MODERATE_EFFECT:
            return f"weakly {sign} - {sign}" if sign == "positive" else f"{sign} - weakly {sign}"
        elif improvement > self.MODERATE_EFFECT and improvement <= self.STRONG_MODERATE_EFFECT:
            return sign
        elif improvement > self.STRONG_MODERATE_EFFECT and improvement <= self.STRONG_EFFECT:
            return f"{sign} - strongly {sign}" if sign == "positive" else f"strongly {sign} - {sign}"
        else:
            return f"strongly {sign}"

    def get_ranges(self) -> Dict[str, tuple]:
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
            "WN-IF": (-self.WEAK_EFFECT, -self.WEAK_INDIFERENT_EFFECT),
            "IF": (-self.WEAK_INDIFERENT_EFFECT, self.WEAK_INDIFERENT_EFFECT),
            "IF-WP": (self.WEAK_INDIFERENT_EFFECT, self.WEAK_EFFECT),
            "WP": (self.WEAK_EFFECT, self.WEAK_MODERATE_EFFECT),
            "WP-PO": (self.WEAK_MODERATE_EFFECT, self.MODERATE_EFFECT),
            "PO": (self.MODERATE_EFFECT, self.STRONG_MODERATE_EFFECT),
            "PO-SP": (self.STRONG_MODERATE_EFFECT, self.STRONG_EFFECT),
            "SP": (self.STRONG_EFFECT, np.inf),
        }


class EnergyIntensity(EffectIntensity):
    @property
    @staticmethod
    def STRONG_EFFECT(self):
        return 50

    @property
    @staticmethod
    def STRONG_MODERATE_EFFECT(self):
        return 40

    @property
    @staticmethod
    def MODERATE_EFFECT(self):
        return 30

    @property
    @staticmethod
    def WEAK_MODERATE_EFFECT(self):
        return 20

    @property
    @staticmethod
    def WEAK_EFFECT(self):
        return 10

    @property
    @staticmethod
    def WEAK_INDIFERENT_EFFECT(self):
        return 2


class ResourceUsageIntensity(EffectIntensity):
    @property
    @staticmethod
    def STRONG_EFFECT(self):
        return 50

    @property
    @staticmethod
    def STRONG_MODERATE_EFFECT(self):
        return 40

    @property
    @staticmethod
    def MODERATE_EFFECT(self):
        return 30

    @property
    @staticmethod
    def WEAK_MODERATE_EFFECT(self):
        return 20

    @property
    @staticmethod
    def WEAK_EFFECT(self):
        return 10

    @property
    @staticmethod
    def WEAK_INDIFERENT_EFFECT(self):
        return 2


class LatencyIntensity(EffectIntensity):
    @property
    @staticmethod
    def STRONG_EFFECT(self):
        return 50

    @property
    @staticmethod
    def STRONG_MODERATE_EFFECT(self):
        return 40

    @property
    @staticmethod
    def MODERATE_EFFECT(self):
        return 30

    @property
    @staticmethod
    def WEAK_MODERATE_EFFECT(self):
        return 20

    @property
    @staticmethod
    def WEAK_EFFECT(self):
        return 10

    @property
    @staticmethod
    def WEAK_INDIFERENT_EFFECT(self):
        return 2


class CorrectnessIntensity(EffectIntensity):
    @property
    @staticmethod
    def STRONG_EFFECT(self):
        return 25

    @property
    @staticmethod
    def STRONG_MODERATE_EFFECT(self):
        return 20

    @property
    @staticmethod
    def MODERATE_EFFECT(self):
        return 15

    @property
    @staticmethod
    def WEAK_MODERATE_EFFECT(self):
        return 10

    @property
    @staticmethod
    def WEAK_EFFECT(self):
        return 5
