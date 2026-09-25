"""In-sample screening performance metrics with uncertainty."""

from src.data.selection.screening_metrics import jeffreys_interval, proportion

CALIBRATION_RECALL_LOW = 0.82908315
CALIBRATION_RECALL_HIGH = 0.98793635
ZERO_OF_HUNDRED_HIGH_MIN = 0.01
ZERO_OF_HUNDRED_HIGH_MAX = 0.03
FLOAT_TOL = 1e-6


def test_proportion_of_calibration_recall():
    assert proportion(33, 35) == 33 / 35


def test_jeffreys_interval_for_calibration_recall_matches_beta_quantiles():
    low, high = jeffreys_interval(33, 35)
    assert abs(low - CALIBRATION_RECALL_LOW) < FLOAT_TOL
    assert abs(high - CALIBRATION_RECALL_HIGH) < FLOAT_TOL


def test_jeffreys_interval_for_zero_of_one_hundred_is_near_zero():
    low, high = jeffreys_interval(0, 100)
    assert low == 0.0
    assert ZERO_OF_HUNDRED_HIGH_MIN < high < ZERO_OF_HUNDRED_HIGH_MAX
