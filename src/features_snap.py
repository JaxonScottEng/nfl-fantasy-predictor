# features_snap.py
"""
Snap share -- how much of his team's offensive snaps a player was on the field for.

Raw target/carry counts conflate role with game flow; snap share is the cleaner
statement of "how much does this offense actually use him". Rolling and lagged
like every other feature.

Join wrinkle: load_snap_counts() carries only `pfr_player_id`, no gsis_id, so it
cannot join to player_id directly (and CLAUDE.md forbids joining on name). We
bridge through load_rosters_weekly, which carries both. That bridge covers ~92%
of WR/RB rows; the rest stay NaN rather than 0, because 0 would falsely assert
"was on the field for no snaps" when the truth is "unknown". XGBoost splits on
NaN natively.
"""
import nflreadpy as nfl
import numpy as np
import pandas as pd

import config
from data_load import cached_table

SNAP_COL = "offense_pct"


def load_snap_counts_cached(force_refresh=False):
    return cached_table(
        "snap_counts",
        lambda: nfl.load_snap_counts(config.SEASONS),
        force_refresh=force_refresh,
    )


def load_pfr_bridge(force_refresh=False):
    """pfr_id -> player_id (gsis) per season, from weekly rosters."""
    rosters = cached_table(
        "rosters_weekly",
        lambda: nfl.load_rosters_weekly(config.SEASONS),
        force_refresh=force_refresh,
    )
    bridge = rosters[["season", "gsis_id", "pfr_id"]].dropna(subset=["gsis_id", "pfr_id"])
    bridge = bridge.rename(columns={"gsis_id": "player_id"})
    return bridge.drop_duplicates(subset=["season", "pfr_id"])


def build_snap_features(windows=None, force_refresh=False, extra_keys=None):
    """
    player_id, season, week + lagged rolling offensive snap share.

    `extra_keys` appends (player_id, season, week) rows with a NaN snap value --
    the same placeholder trick upcoming.py uses, so a not-yet-played week gets a
    properly lagged rolling value instead of a missing join.
    """
    windows = windows or config.ROLLING_WINDOWS

    snaps = load_snap_counts_cached(force_refresh=force_refresh)
    snaps = snaps[snaps["game_type"] == "REG"]
    snaps = snaps[["season", "week", "pfr_player_id", SNAP_COL]].copy()
    snaps = snaps.rename(columns={"pfr_player_id": "pfr_id"})

    bridge = load_pfr_bridge(force_refresh=force_refresh)
    snaps = snaps.merge(bridge, on=["season", "pfr_id"], how="inner")
    snaps = snaps[["player_id", "season", "week", SNAP_COL]]

    if extra_keys is not None and len(extra_keys):
        placeholders = extra_keys[["player_id", "season", "week"]].drop_duplicates().copy()
        placeholders[SNAP_COL] = np.nan
        snaps = pd.concat([snaps, placeholders], ignore_index=True)

    snaps = snaps.sort_values(["player_id", "season", "week"])
    snaps[SNAP_COL] = snaps[SNAP_COL].astype(float)

    for w in windows:
        snaps[f"{SNAP_COL}_roll{w}"] = (
            snaps.groupby("player_id")[SNAP_COL]
                 .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )

    keep = ["player_id", "season", "week"] + [f"{SNAP_COL}_roll{w}" for w in windows]
    return snaps[keep].drop_duplicates(subset=["player_id", "season", "week"])


SNAP_FEATURE_COLS = [f"{SNAP_COL}_roll{w}" for w in config.ROLLING_WINDOWS]


if __name__ == "__main__":
    df = build_snap_features()
    print("shape:", df.shape)
    print("null rate:")
    print(df[SNAP_FEATURE_COLS].isna().mean().round(4).to_string())
    print()
    print(df.sort_values(["player_id", "season", "week"]).head(8).to_string(index=False))
