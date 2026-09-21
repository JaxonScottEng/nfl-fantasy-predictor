"""
Feature lag correctness -- the other half of the leakage story.

walk_forward_folds keeps future ROWS out of training; these tests keep future
VALUES out of a row's own features. A rolling mean that accidentally includes week
W's own value would leak the outcome into its own predictor, and every metric would
look wonderful.
"""
import numpy as np
import pandas as pd
import pytest

from features_player import add_rolling_player_features

PLAYER_A = "00-0000001"
PLAYER_B = "00-0000002"


@pytest.fixture
def rolled(tiny_weekly):
    return add_rolling_player_features(tiny_weekly()).sort_values(
        ["player_id", "season", "week"]
    ).reset_index(drop=True)


def _value(frame, player, week, column):
    row = frame[(frame["player_id"] == player) & (frame["week"] == week)]
    assert len(row) == 1
    return row.iloc[0][column]


def test_first_game_has_no_history(rolled):
    assert np.isnan(_value(rolled, PLAYER_A, 1, "targets_roll3"))


def test_rolling_mean_uses_only_prior_rows(rolled):
    """Player A's targets are 1,2,3,4,5,6 -- week 4's roll3 is mean(1,2,3) = 2.0."""
    assert _value(rolled, PLAYER_A, 4, "targets_roll3") == pytest.approx(2.0)
    assert _value(rolled, PLAYER_A, 5, "targets_roll3") == pytest.approx(3.0)
    # Week 2 sees one prior game only (min_periods=1).
    assert _value(rolled, PLAYER_A, 2, "targets_roll3") == pytest.approx(1.0)


def test_current_week_value_is_excluded(rolled):
    """The whole point: week 6's window must not contain week 6's own value (6.0)."""
    week_6 = _value(rolled, PLAYER_A, 6, "targets_roll3")
    assert week_6 == pytest.approx(4.0)       # mean(3, 4, 5)
    assert week_6 != pytest.approx(5.0)       # what roll-then-shift would give


def test_windows_do_not_cross_players(rolled):
    """Player B's values are 100x larger; any bleed would be obvious."""
    for week in range(1, 7):
        value = _value(rolled, PLAYER_A, week, "targets_roll5")
        assert np.isnan(value) or value < 10.0


def test_ratio_features_are_built_from_lagged_columns(rolled):
    """
    yards_per_target must equal lagged_yards / lagged_targets. Building it from raw
    stats and lagging afterwards gives a different number and leaks the current week.
    """
    for week in range(2, 7):
        ratio = _value(rolled, PLAYER_A, week, "yards_per_target_roll4")
        yards = _value(rolled, PLAYER_A, week, "receiving_yards_roll4")
        targets = _value(rolled, PLAYER_A, week, "targets_roll4")
        assert ratio == pytest.approx(yards / targets)


def test_zero_denominator_yields_float_nan_not_object_dtype():
    """
    Pins a real bug from Phase 4: pd.NA in a ratio produces object dtype, which
    XGBoost rejects outright. np.nan keeps the column float64.
    """
    from features_player import ROLLING_INPUT_COLS

    rows = []
    for week in range(1, 5):
        row = {
            "player_id": PLAYER_A, "player_display_name": "A", "position": "WR",
            "season": 2020, "week": week, "team": "SF", "opponent_team": "LA",
            "fantasy_points_ppr": 0.0,
        }
        for col in ROLLING_INPUT_COLS:
            row[col] = 0.0          # zero targets -> zero-denominator ratio
        rows.append(row)

    out = add_rolling_player_features(pd.DataFrame(rows))

    assert out["yards_per_target_roll3"].dtype == np.float64
    assert out["yards_per_target_roll3"].isna().all()


def test_touches_is_the_sum_of_its_lagged_parts(rolled):
    for week in range(2, 7):
        touches = _value(rolled, PLAYER_A, week, "touches_roll3")
        carries = _value(rolled, PLAYER_A, week, "carries_roll3")
        receptions = _value(rolled, PLAYER_A, week, "receptions_roll3")
        assert touches == pytest.approx(carries + receptions)
