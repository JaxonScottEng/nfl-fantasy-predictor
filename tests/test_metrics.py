"""Metric correctness, including the bias sign convention."""
import numpy as np
import pandas as pd
import pytest

from metrics import METRICS

Y_TRUE = np.array([10.0, 20.0, 30.0, 40.0])
Y_PRED = np.array([12.0, 18.0, 33.0, 39.0])          # errors +2, -2, +3, -1


def test_mae_matches_hand_computation():
    assert METRICS.get("mae")(Y_TRUE, Y_PRED) == pytest.approx(2.0)


def test_rmse_matches_hand_computation():
    expected = np.sqrt((4 + 4 + 9 + 1) / 4)
    assert METRICS.get("rmse")(Y_TRUE, Y_PRED) == pytest.approx(expected)


def test_rmse_is_never_below_mae():
    assert METRICS.get("rmse")(Y_TRUE, Y_PRED) >= METRICS.get("mae")(Y_TRUE, Y_PRED)


def test_r2_is_one_for_a_perfect_fit():
    assert METRICS.get("r2")(Y_TRUE, Y_TRUE) == pytest.approx(1.0)


def test_r2_is_zero_when_predicting_the_mean():
    mean_pred = np.full_like(Y_TRUE, Y_TRUE.mean())
    assert METRICS.get("r2")(Y_TRUE, mean_pred) == pytest.approx(0.0)


def test_r2_goes_negative_when_worse_than_the_mean():
    """Not a curiosity: the naive baseline scores negative R^2 on the top-40 pool."""
    awful = np.array([100.0, -50.0, 200.0, -30.0])
    assert METRICS.get("r2")(Y_TRUE, awful) < 0


def test_mean_error_sign_convention_is_negative_for_under_projection():
    """
    Pins the direction behind the -4.4 elite-player finding. mean_error is
    pred - actual, so a model projecting too low must report a NEGATIVE number.
    """
    under = Y_TRUE - 5.0
    over = Y_TRUE + 5.0

    assert METRICS.get("mean_error")(Y_TRUE, under) == pytest.approx(-5.0)
    assert METRICS.get("mean_error")(Y_TRUE, over) == pytest.approx(5.0)


def test_group_metrics_require_groups():
    for metric_id in ("spearman_within", "top_n_hit_rate", "cov_weekly_mae"):
        with pytest.raises(ValueError, match="groups"):
            METRICS.get(metric_id)(Y_TRUE, Y_PRED)


def test_spearman_is_one_for_perfectly_ordered_predictions():
    groups = pd.Series(["w1"] * 4)
    scaled = Y_TRUE * 3.0 + 1.0          # different scale, identical ordering
    assert METRICS.get("spearman_within")(Y_TRUE, scaled, groups=groups) == pytest.approx(1.0)


def test_top_n_hit_rate_is_one_when_the_top_n_match():
    groups = pd.Series(["w1"] * 4)
    rate = METRICS.get("top_n_hit_rate")(Y_TRUE, Y_PRED, groups=groups, n=2)
    assert rate == pytest.approx(1.0)     # both rank 30/40 highest


def test_cov_weekly_mae_is_zero_for_identical_weeks():
    groups = pd.Series(["w1", "w1", "w2", "w2"])
    y_true = np.array([10.0, 20.0, 10.0, 20.0])
    y_pred = np.array([11.0, 21.0, 11.0, 21.0])
    assert METRICS.get("cov_weekly_mae")(y_true, y_pred, groups=groups) == pytest.approx(0.0)
