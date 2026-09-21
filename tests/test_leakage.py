"""
Leakage checks: nothing may see data from the week it is predicting.

These are the rules every accuracy number depends on. If they fail, the numbers
are meaningless.
"""
from data_load import filter_regular_season
from features_player import add_rolling_player_features
from validation import walk_forward_folds
import numpy as np
import pandas as pd
import pytest


# --- from test_validation ---

TRAIN_SEASON = 2020
VALIDATION_SEASON = 2021


@pytest.fixture
def two_seasons():
    rows = [
        {"season": season, "week": week, "row_id": f"{season}-{week}"}
        for season in (TRAIN_SEASON, VALIDATION_SEASON)
        for week in range(1, 7)
    ]
    return pd.DataFrame(rows)


def test_no_train_row_is_from_the_test_week_or_later(two_seasons):
    for week, train_df, _ in walk_forward_folds(two_seasons, VALIDATION_SEASON):
        leaked = train_df[(train_df["season"] == VALIDATION_SEASON)
                          & (train_df["week"] >= week)]
        assert len(leaked) == 0, f"week {week} trained on week {week} or later"


def test_folds_advance_within_the_season(two_seasons):
    """Week 5 must train on validation weeks 1-4, not just on prior seasons."""
    folds = {week: train for week, train, _ in walk_forward_folds(two_seasons, VALIDATION_SEASON)}

    in_season = folds[5][folds[5]["season"] == VALIDATION_SEASON]
    assert sorted(in_season["week"]) == [1, 2, 3, 4]


def test_every_validation_week_is_tested_exactly_once(two_seasons):
    tested = [week for week, _, _ in walk_forward_folds(two_seasons, VALIDATION_SEASON)]
    assert tested == sorted(tested)
    assert len(tested) == len(set(tested))
    # Week 1 has prior-season training data, so all six weeks are testable.
    assert tested == [1, 2, 3, 4, 5, 6]


def test_test_fold_holds_only_the_target_week(two_seasons):
    for week, _, test_df in walk_forward_folds(two_seasons, VALIDATION_SEASON):
        assert set(test_df["week"]) == {week}
        assert set(test_df["season"]) == {VALIDATION_SEASON}


def test_fold_with_no_training_data_is_skipped():
    """The validation season's first week has nothing before it -- yield nothing."""
    only_validation = pd.DataFrame(
        [{"season": VALIDATION_SEASON, "week": w} for w in range(1, 4)]
    )
    weeks = [week for week, _, _ in walk_forward_folds(only_validation, VALIDATION_SEASON)]
    assert 1 not in weeks


def test_validation_season_argument_is_honored(two_seasons):
    """Pointing the generator at an earlier season tests that season's weeks."""
    folds = list(walk_forward_folds(two_seasons, TRAIN_SEASON))
    assert folds, "explicit validation_season should produce folds"

    for week, train_df, test_df in folds:
        assert set(test_df["season"]) == {TRAIN_SEASON}
        # Training may include earlier weeks of the same season -- that is the
        # walk advancing -- but never the target week or later.
        assert train_df[(train_df["season"] == TRAIN_SEASON)
                        & (train_df["week"] >= week)].empty
        assert (train_df["season"] <= TRAIN_SEASON).all()


# --- from test_features_lag ---

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
    Pins a real bug: pd.NA in a ratio produces object dtype, which
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


# --- from test_postseason ---

def test_filter_drops_postseason_rows(tiny_weekly):
    regular = tiny_weekly(season_type="REG")
    playoff = tiny_weekly(season_type="POST", weeks=range(19, 22))
    combined = pd.concat([regular, playoff], ignore_index=True)

    out = filter_regular_season(combined)

    assert set(out["season_type"]) == {"REG"}
    assert len(out) == len(regular)


def test_filter_returns_a_copy_not_a_view(tiny_weekly):
    """A view would raise SettingWithCopyWarning downstream when columns are added."""
    out = filter_regular_season(tiny_weekly())
    out["scratch"] = 1
    assert "scratch" in out.columns


def test_playoff_values_never_reach_a_rolling_window(monkeypatch, tiny_weekly):
    """
    The end-to-end property. Regular-season targets are small; playoff rows carry a
    huge sentinel. If any rolling feature picks up the sentinel, the filter ran too
    late in the pipeline.
    """
    import features_player

    regular = tiny_weekly(weeks=range(1, 6), season_type="REG")
    playoff = tiny_weekly(weeks=range(19, 21), season_type="POST")
    for col in ["targets", "receptions", "receiving_yards"]:
        playoff[col] = 10_000.0
    combined = pd.concat([regular, playoff], ignore_index=True)

    monkeypatch.setattr(features_player, "load_weekly", lambda: combined)
    out = features_player.build_player_features()

    assert out["week"].max() <= 18
    for column in ["targets_roll3", "targets_roll5", "receiving_yards_roll4"]:
        assert (out[column].dropna() < 1_000.0).all(), f"{column} saw a playoff value"
