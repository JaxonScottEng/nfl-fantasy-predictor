# features_expected.py
"""
Expected fantasy points from nflverse: what a player's opportunities were worth,
rather than what he scored.

A receiver who drew 11 targets and 140 air yards had a good week of opportunity even
if he caught two balls. Expected points are more stable week to week than actual
points, because touchdowns are close to random in the short term.

Only lagged values are produced. Expected points for week W come from week W's
plays, so a same-week value would leak the outcome.
"""
import nflreadpy as nfl
import numpy as np
import pandas as pd

import config
from data_load import cached_table

EXPECTED_COLS = [
    "total_fantasy_points_exp",
    "rec_fantasy_points_exp",
    "rush_fantasy_points_exp",
    "receptions_exp",
    "total_touchdown_exp",
]


def load_ff_opportunity_cached(force_refresh=False):
    """Weekly expected points. `season` arrives as String and `week` as float."""
    df = cached_table(
        "ff_opportunity",
        lambda: nfl.load_ff_opportunity(config.SEASONS, stat_type="weekly"),
        force_refresh=force_refresh,
    )
    df = df.copy()
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    return df


def build_expected_points_features(windows=None, force_refresh=False, extra_keys=None):
    """
    player_id, season, week + lagged rolling expected-production features.

    `extra_keys` appends (player_id, season, week) rows with NaN values -- the same
    placeholder trick upcoming.py uses, so a not-yet-played week gets properly
    lagged rolling values instead of a missing join.
    """
    windows = windows or config.ROLLING_WINDOWS

    df = load_ff_opportunity_cached(force_refresh=force_refresh)
    keep = ["player_id", "season", "week"] + EXPECTED_COLS
    df = df[keep].dropna(subset=["player_id"]).copy()
    df = df.drop_duplicates(subset=["player_id", "season", "week"])

    if extra_keys is not None and len(extra_keys):
        placeholders = extra_keys[["player_id", "season", "week"]].drop_duplicates().copy()
        for col in EXPECTED_COLS:
            placeholders[col] = np.nan
        df = pd.concat([df, placeholders], ignore_index=True)
        df = df.drop_duplicates(subset=["player_id", "season", "week"], keep="first")

    df = df.sort_values(["player_id", "season", "week"])

    for col in EXPECTED_COLS:
        df[col] = df[col].astype(float)
        for w in windows:
            df[f"{col}_roll{w}"] = (
                df.groupby("player_id")[col]
                  .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )

    rolled = [f"{col}_roll{w}" for col in EXPECTED_COLS for w in windows]
    return df[["player_id", "season", "week"] + rolled]


EXPECTED_FEATURE_COLS = [
    f"{col}_roll{w}" for col in EXPECTED_COLS for w in config.ROLLING_WINDOWS
]

# The comparator in comparators.py projects with this one.
EXPECTED_BASELINE_COL = "total_fantasy_points_exp_roll4"


if __name__ == "__main__":
    df = build_expected_points_features()
    print("shape:", df.shape)
    print("\nnull rates:")
    print(df[EXPECTED_FEATURE_COLS].isna().mean().round(4).to_string())
    print("\nsample:")
    print(df[["player_id", "season", "week", "total_fantasy_points_exp_roll4",
              "total_touchdown_exp_roll4"]].head(8).to_string(index=False))
