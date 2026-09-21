"""
The leakage invariant.

This assertion previously lived only in validation.py's `__main__`, which means it
ran when someone happened to execute the file. It is the project's single most
important property -- every accuracy number is meaningless if it fails -- so it
belongs in a suite that runs on every change.
"""
import pandas as pd
import pytest

from validation import walk_forward_folds

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
