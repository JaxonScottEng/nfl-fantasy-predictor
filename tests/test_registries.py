"""
The registries and the scoring functions they hold.

Covers the player pools, the accuracy metrics, the quantile scores, and the
registry itself, plus a check that the ids other modules depend on still exist.
"""
from metrics import METRICS
from model_quantile import QUANTILE_ALPHAS, interval_coverage, pinball_loss
from pools import POOLS, DEFAULT_POOL_SIZE
from registry import Registry
import numpy as np
import pandas as pd
import pytest


# --- from test_registry ---

def test_register_and_retrieve():
    reg = Registry("widget")

    @reg.register("alpha", label="Alpha", caveat="approximate", size=3)
    def alpha():
        return "a"

    assert reg.get("alpha")() == "a"
    assert reg.available() == ["alpha"]
    assert reg.label("alpha") == "Alpha"
    assert reg.caveat("alpha") == "approximate"
    assert reg.meta("alpha", "size") == 3
    assert "alpha" in reg
    assert len(reg) == 1


def test_duplicate_id_is_rejected():
    reg = Registry("widget")
    reg.register("alpha")(lambda: None)

    with pytest.raises(ValueError, match="already registered"):
        reg.register("alpha")(lambda: None)


def test_unknown_id_lists_what_is_available():
    reg = Registry("widget")
    reg.register("alpha")(lambda: None)

    with pytest.raises(KeyError) as excinfo:
        reg.get("missing")
    assert "alpha" in str(excinfo.value)


def test_label_defaults_to_the_id():
    reg = Registry("widget")
    reg.register("alpha")(lambda: None)
    assert reg.label("alpha") == "alpha"


# --- integration guards -------------------------------------------------------

def test_shipped_comparator_resolves():
    from comparators import COMPARATORS, SHIPPED_COMPARATOR_ID
    assert SHIPPED_COMPARATOR_ID in COMPARATORS


def test_default_comparator_ids_all_resolve():
    from comparators import COMPARATORS, DEFAULT_COMPARATOR_IDS, ALL_POINT_COMPARATOR_IDS
    for comparator_id in set(DEFAULT_COMPARATOR_IDS) | set(ALL_POINT_COMPARATOR_IDS):
        assert COMPARATORS.get(comparator_id) is not None


def test_default_metric_ids_all_resolve():
    from metrics import METRICS, DEFAULT_METRIC_IDS
    for metric_id in DEFAULT_METRIC_IDS:
        assert METRICS.get(metric_id) is not None


def test_gui_range_methods_resolve():
    """The GUI builds its selector from available_methods(); a rename breaks it."""
    from gui import prediction_ranges

    methods = prediction_ranges.available_methods()
    assert methods, "at least one range method must be registered"
    for method in methods:
        assert prediction_ranges.METHOD_LABELS.get(method)
        assert prediction_ranges.METHOD_CAVEATS.get(method)


def test_hook_expectations_cover_every_active_position():
    """
    The regression hook hardcodes the locked-in numbers. If a position is added to
    config without updating the hook, drift stops being detected for it.
    """
    import importlib.util
    import os

    import config
    from conftest import PROJECT_ROOT

    hook_path = os.path.join(PROJECT_ROOT, ".claude", "hooks", "check_wr_regression.py")
    spec = importlib.util.spec_from_file_location("check_wr_regression", hook_path)
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)

    for position in config.ACTIVE_POSITIONS:
        assert position in hook.EXPECTED_MAE, f"hook has no expectations for {position}"


# --- from test_pools ---

@pytest.fixture
def week_frame():
    return pd.DataFrame({
        "player_id": [f"p{i}" for i in range(10)],
        "_pred": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        # Deliberately anti-correlated so top-by-projection and top-by-actual differ.
        "_actual": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })


def test_all_rows_returns_everything(week_frame):
    selected = POOLS.get("all")(week_frame)
    assert len(selected) == len(week_frame)


def test_top_n_by_projection_takes_the_n_largest(week_frame):
    selected = POOLS.get("top_n_by_projection")(week_frame, pred_col="_pred", n=3)
    assert set(week_frame.loc[selected, "_pred"]) == {10, 9, 8}


def test_top_n_handles_n_larger_than_the_frame(week_frame):
    selected = POOLS.get("top_n_by_projection")(week_frame, pred_col="_pred", n=99)
    assert len(selected) == len(week_frame)


def test_union_top_n_is_a_superset_of_both_halves(week_frame):
    n = 3
    union = POOLS.get("union_top_n")(week_frame, pred_col="_pred", actual_col="_actual", n=n)
    by_pred = POOLS.get("top_n_by_projection")(week_frame, pred_col="_pred", n=n)

    assert set(by_pred).issubset(set(union))
    assert n <= len(union) <= 2 * n


def test_every_pool_declares_a_caveat():
    """
    A pool without a stated caveat is how a number quietly becomes incomparable
    again. Registering one without an explanation should fail here.
    """
    for pool_id in POOLS.available():
        caveat = POOLS.caveat(pool_id)
        assert caveat and caveat.strip(), f"pool {pool_id!r} has no caveat"


def test_published_study_pool_sizes_are_declared():
    """Matches the convention published accuracy studies use."""
    assert DEFAULT_POOL_SIZE["WR"] == 40
    assert DEFAULT_POOL_SIZE["RB"] == 40
    assert DEFAULT_POOL_SIZE["QB"] == 20
    assert DEFAULT_POOL_SIZE["TE"] == 20


# --- from test_metrics ---

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


# --- from test_quantile_scoring ---

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
