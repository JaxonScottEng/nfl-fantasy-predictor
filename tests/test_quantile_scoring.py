"""
Quantile scoring.

These two functions produced the 0.78/0.79 interval-coverage figures quoted in
CLAUDE.md but had no caller, so the published number was not reproducible by
running anything. Testing them makes the claim checkable.
"""
import numpy as np
import pytest

from model_quantile import QUANTILE_ALPHAS, interval_coverage, pinball_loss

Y_TRUE = np.array([10.0, 20.0, 30.0, 40.0])
Y_PRED = np.array([12.0, 18.0, 33.0, 39.0])


def test_pinball_at_the_median_is_half_the_mae():
    """An identity worth pinning: it is why q50 optimizes MAE directly."""
    mae = np.mean(np.abs(Y_TRUE - Y_PRED))
    assert pinball_loss(Y_TRUE, Y_PRED, 0.5) == pytest.approx(0.5 * mae)


def test_pinball_is_zero_for_perfect_predictions():
    for alpha in QUANTILE_ALPHAS:
        assert pinball_loss(Y_TRUE, Y_TRUE, alpha) == pytest.approx(0.0)


def test_high_alpha_punishes_under_prediction_harder():
    """A 90th-percentile estimate should rather overshoot than undershoot."""
    under = Y_TRUE - 5.0
    over = Y_TRUE + 5.0
    assert pinball_loss(Y_TRUE, under, 0.9) > pinball_loss(Y_TRUE, over, 0.9)


def test_low_alpha_punishes_over_prediction_harder():
    under = Y_TRUE - 5.0
    over = Y_TRUE + 5.0
    assert pinball_loss(Y_TRUE, over, 0.1) > pinball_loss(Y_TRUE, under, 0.1)


def test_coverage_counts_actuals_inside_the_interval():
    y_true = np.arange(10, dtype=float)
    lo = np.full(10, 2.0)
    hi = np.full(10, 9.0)
    assert interval_coverage(y_true, lo, hi) == pytest.approx(0.8)


def test_coverage_bounds_are_inclusive():
    y_true = np.array([5.0])
    assert interval_coverage(y_true, np.array([5.0]), np.array([9.0])) == pytest.approx(1.0)
    assert interval_coverage(y_true, np.array([1.0]), np.array([5.0])) == pytest.approx(1.0)


def test_coverage_is_zero_when_nothing_lands_inside():
    y_true = np.array([100.0, 200.0])
    assert interval_coverage(y_true, np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0


def test_quantile_alphas_are_ordered_and_symmetric():
    assert list(QUANTILE_ALPHAS) == sorted(QUANTILE_ALPHAS)
    assert QUANTILE_ALPHAS[0] + QUANTILE_ALPHAS[-1] == pytest.approx(1.0)
