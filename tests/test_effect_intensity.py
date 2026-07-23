from src.effect_intensity import CorrectnessIntensity, EffectIntensity


def test_effect_intensity_boundaries():
    intensity = EffectIntensity()

    assert intensity.get_intensity(0) == "indiferent"
    assert intensity.get_intensity(2) == "indiferent"
    assert intensity.get_intensity(5) == "indiferent - weakly positive"
    assert intensity.get_intensity(-5) == "weakly negative - indiferent"
    assert intensity.get_intensity(55) == "strongly positive"
    assert intensity.get_intensity(-55) == "strongly negative"

def test_correctness_intensity_uses_tighter_thresholds():
    intensity = CorrectnessIntensity()

    assert intensity.get_intensity(6) == "weakly positive"
    assert intensity.get_intensity(26) == "strongly positive"
