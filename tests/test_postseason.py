"""
Postseason exclusion.

Phase 9 found playoff games sitting inside training data AND inside rolling
windows, because MAX_VALIDATION_WEEK only trimmed the validation season. Playoff
games have a different player pool (14 teams, rested starters), so mixing them in
compares unlike things. Filtering only the prediction targets is not enough -- the
filter has to run before any rolling, which is what these tests pin.
"""
import pandas as pd
import pytest

from data_load import filter_regular_season


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


def test_target_table_excludes_postseason(monkeypatch, tiny_weekly):
    import build_target

    regular = tiny_weekly(weeks=range(1, 6), season_type="REG")
    playoff = tiny_weekly(weeks=range(19, 21), season_type="POST")
    combined = pd.concat([regular, playoff], ignore_index=True)

    monkeypatch.setattr(build_target, "load_weekly", lambda: combined)
    out = build_target.get_target_table()

    assert out["week"].max() <= 18
