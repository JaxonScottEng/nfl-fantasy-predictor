# player_timeline.py
"""
One position's season week by week: what was projected, the range around it, and
what actually happened.

Played weeks come from the walk-forward evaluation, so each prediction used only
earlier games. Unplayed weeks come from upcoming.predict_weeks. The boundary between
them is reported in the metadata so the app can mark it.
"""
import numpy as np
import pandas as pd

import config
from build_features import build_full_feature_table
from comparators import SHIPPED_COMPARATOR_ID
from evaluate import run_evaluation
from features_context import load_schedules_cached
from upcoming import predict_weeks

TIMELINE_COMPARATORS = [SHIPPED_COMPARATOR_ID, "baseline_last4", "xgb_q10", "xgb_q90"]

OUTPUT_COLS = ["season", "week", "player_id", "player_display_name", "opponent_team",
               "actual", "ensemble_pred", "baseline_pred", "q10", "q90", "is_played"]


def season_week_status(season, feature_table):
    """
    (played_weeks, unplayed_weeks) for a season.

    "Played" is derived from the PLAYER data, not the schedule, deliberately. The two
    caches refresh independently, and when the schedule is newer it reports a week as
    complete that we have no player rows for -- that week would then be neither
    scored nor projected, and would silently vanish from the timeline. Keying off the
    player data instead means a week we cannot score is simply projected, which is
    visible and correct. The schedule still defines which weeks exist at all.
    """
    sched = load_schedules_cached()
    sched = sched[(sched["season"] == season) & (sched["week"] <= config.MAX_VALIDATION_WEEK)]
    if len(sched) == 0:
        return [], []

    scheduled = sorted(int(w) for w in sched["week"].unique())
    scored = feature_table[(feature_table["season"] == season)
                           & feature_table[config.TARGET].notna()]
    played = sorted(int(w) for w in scored["week"].unique())
    unplayed = [w for w in scheduled if w not in set(played)]
    return played, unplayed


def _played_timeline(position, season, played_weeks, feature_table):
    if not played_weeks:
        return pd.DataFrame()

    rows = run_evaluation(positions=[position], seasons=[season], pool="all",
                          comparator_ids=TIMELINE_COMPARATORS,
                          feature_table=feature_table, verbose=False)
    if len(rows) == 0:
        return pd.DataFrame()

    wide = rows.pivot_table(
        index=["season", "week", "player_id", "player_display_name", "actual"],
        columns="comparator", values="pred",
    ).reset_index()

    wide = wide.rename(columns={SHIPPED_COMPARATOR_ID: "ensemble_pred",
                                "baseline_last4": "baseline_pred",
                                "xgb_q10": "q10", "xgb_q90": "q90"})

    # The harness returns predictions, not matchups; opponent comes from the table.
    opponents = feature_table[["player_id", "season", "week", "opponent_team"]]
    wide = wide.merge(opponents, on=["player_id", "season", "week"], how="left")
    wide["is_played"] = True
    return wide


def _outlook_timeline(position, season, unplayed_weeks):
    if not unplayed_weeks:
        return pd.DataFrame()

    rows = predict_weeks(position, season, unplayed_weeks)
    if len(rows) == 0:
        return pd.DataFrame()

    rows = rows.copy()
    rows["actual"] = np.nan
    rows["is_played"] = False
    return rows


def build_position_timeline(position, season):
    """
    Long timeline for every player at `position` in `season`. Returns (df, meta).
    Weeks a player missed simply have no row -- there is nothing to project from
    and nothing to plot.
    """
    feature_table = build_full_feature_table()
    played_weeks, unplayed_weeks = season_week_status(season, feature_table)

    played = _played_timeline(position, season, played_weeks, feature_table)
    outlook = _outlook_timeline(position, season, unplayed_weeks)

    frames = [f for f in (played, outlook) if len(f)]
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLS), {"season": season}

    combined = pd.concat(frames, ignore_index=True)
    for col in OUTPUT_COLS:
        if col not in combined.columns:
            combined[col] = np.nan
    combined = combined[OUTPUT_COLS].sort_values(["player_display_name", "week"])

    meta = {
        "season": season,
        "played_weeks": played_weeks,
        "unplayed_weeks": unplayed_weeks,
        "first_unplayed_week": unplayed_weeks[0] if unplayed_weeks else None,
        # Weeks past the next one, where the projection is a form outlook only.
        "outlook_weeks": unplayed_weeks[1:] if len(unplayed_weeks) > 1 else [],
    }
    return combined.reset_index(drop=True), meta


def build_player_timeline(position, season, player_display_name):
    df, meta = build_position_timeline(position, season)
    if len(df) == 0:
        return df, meta
    return (df[df["player_display_name"] == player_display_name]
            .sort_values("week").reset_index(drop=True), meta)


if __name__ == "__main__":
    season = 2026
    df, meta = build_position_timeline("WR", season)
    print(f"{season}: played weeks {meta['played_weeks']}, "
          f"first unplayed {meta['first_unplayed_week']}")
    print("rows:", len(df), "| players:", df["player_display_name"].nunique())
    print("meta:", {k: v for k, v in meta.items() if k != "outlook_weeks"})
    who = "Puka Nacua"
    sub = df[df["player_display_name"] == who]
    print(f"\n{who}:")
    print(sub[["week", "opponent_team", "actual", "ensemble_pred", "q10", "q90", "is_played"]]
          .head(12).to_string(index=False))
