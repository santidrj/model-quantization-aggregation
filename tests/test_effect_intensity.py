from src.effect_intensity import (
    CorrectnessIntensity,
    EffectIntensity,
    intensity_label_to_code,
    render_intensity_thresholds_table,
    signed_intensity_intervals,
)


def test_effect_intensity_boundaries():
    intensity = EffectIntensity()

    assert intensity.get_intensity(0) == "indifferent"
    assert intensity.get_intensity(2) == "indifferent"
    assert intensity.get_intensity(5) == "indifferent - weakly positive"
    assert intensity.get_intensity(-5) == "weakly negative - indifferent"
    assert intensity.get_intensity(55) == "strongly positive"
    assert intensity.get_intensity(-55) == "strongly negative"


def test_correctness_intensity_uses_tighter_thresholds():
    intensity = CorrectnessIntensity()

    assert intensity.get_intensity(6) == "weakly positive"
    assert intensity.get_intensity(26) == "strongly positive"


def test_signed_intervals_match_get_intensity():
    samples = [-55.0, -50.0, -40.0, -25.0, -2.0, 0.0, 2.0, 2.1, 10.0, 25.0, 50.0, 50.1]
    for scale in (EffectIntensity(), CorrectnessIntensity()):
        intervals = signed_intensity_intervals(scale)
        for value in samples:
            matches = [interval.code for interval in intervals if interval.contains(value)]
            assert matches == [intensity_label_to_code(scale.get_intensity(value))]


def test_intensity_thresholds_table_lists_both_scales():
    latex = render_intensity_thresholds_table()
    assert r"\{SN\}" in latex
    assert r"\{IF, WP\}" in latex
    assert r"$(-\infty,-25)$" in latex
    assert r"$(-\infty,-50)$" in latex
    assert r"$[-2,2]$" in latex
    assert r"$(50,\infty)$" in latex
    assert r"$(25,\infty)$" in latex
