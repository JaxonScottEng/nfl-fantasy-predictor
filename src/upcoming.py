# upcoming.py
"""
Predictions for a week that has NOT been played yet.

The trick that keeps this leakage-free: the existing feature builders all lag
via .shift(1).rolling(...) grouped by player_id. So if we append placeholder
rows for the upcoming week -- carrying only identity/schedule fields, with NaN
for every raw stat and the target -- and run those same builders, the upcoming
row's features are computed strictly from games BEFORE it. The NaN stats in the
placeholder row never feed its own features.

That means zero duplicated rolling logic: this module assembles rows and calls
features_player / features_opponent / features_context / baselines as-is.

The model is trained on completed rows strictly before the upcoming week, which
is the same rule walk_forward_folds enforces for validation.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import nflreadpy as nfl

import config
from data_load import load_weekly, filter_regular_season
from features_player import add_rolling_player_features, ROLLING_INPUT_COLS
from features_opponent import build_opponent_defense_features
from features_snap import build_snap_features
from features_expected import build_expected_points_features
from features_context import load_schedule_context, normalize_team_codes
from baselines import add_rolling_baseline
from train import FEATURE_COLS_BY_POSITION, HYBRID_THRESHOLD_BY_POSITION, DEFAULT_HYBRID_THRESHOLD

# A player needs at least this many completed games in the current season to be
# treated as active. Filters out last year's roster and long-term absences.
MIN_RECENT_GAMES = 1


def find_upcoming_week(schedule=None):
    """
    (season, week) of the next unplayed game, or None if every configured
    season is complete. "Unplayed" = no result recorded yet.
    """
    sched = schedule if schedule is not None else nfl.load_schedules(config.SEASONS).to_pandas()
    unplayed = sched[sched["home_score"].isna()]
    if len(unplayed) == 0:
        return None

    season = int(unplayed["season"].min())
    week = int(unplayed.loc[unplayed["season"] == season, "week"].min())
    return season, week


def _upcoming_matchups(season, week):
    """One row per team playing that week: team + opponent_team."""
    sched = nfl.load_schedules([season]).to_pandas()
    games = sched[sched["week"] == week][["home_team", "away_team"]].copy()
    games = normalize_team_codes(games, ["home_team", "away_team"])

    home = games.rename(columns={"home_team": "team", "away_team": "opponent_team"})
    away = games.rename(columns={"away_team": "team", "home_team": "opponent_team"})
    return pd.concat([home, away], ignore_index=True)


def _placeholder_player_rows(weekly, position, season, week, matchups):
    """
    Identity + schedule fields for each active player at `position`, with NaN
    for every stat. These are the rows we want predictions for.
    """
    season_rows = weekly[(weekly["season"] == season) & (weekly["position"] == position)]
    if len(season_rows) == 0:
        return pd.DataFrame()

    games_played = season_rows.groupby("player_id").size()
    active_ids = games_played[games_played >= MIN_RECENT_GAMES].index

    # Most recent completed game per player gives current team + display name.
    latest = (
        season_rows[season_rows["player_id"].isin(active_ids)]
        .sort_values(["player_id", "week"])
        .groupby("player_id")
        .tail(1)
    )

    rows = latest[["player_id", "player_display_name", "position", "team"]].copy()
    rows = normalize_team_codes(rows, ["team"])
    rows["season"] = season
    rows["week"] = week

    # Inner join drops players whose team is on bye this week.
    rows = rows.merge(matchups, on="team", how="inner")

    for col in ROLLING_INPUT_COLS + [config.TARGET]:
        rows[col] = np.nan

    return rows


def _placeholder_defense_rows(allowed, season, week, matchups):
    """
    Same placeholder trick for the opponent-defense table, so the upcoming
    week's def_points_allowed_roll* is the properly lagged value instead of
    a missing join.
    """
    defenses = matchups[["team"]].rename(columns={"team": "defense_team"}).drop_duplicates()
    positions = allowed["position"].unique()

    rows = defenses.merge(pd.DataFrame({"position": positions}), how="cross")
    rows["season"] = season
    rows["week"] = week
    rows["points_allowed_to_position"] = np.nan
    return rows


def build_upcoming_feature_table(position, season, week):
    """
    Feature table containing ONLY the upcoming week's rows for `position`,
    with every feature computed from strictly-earlier games.
    """
    # Regular season only, matching the training pipeline -- a player's rolling
    # window must mean the same thing here as it does during training.
    weekly = filter_regular_season(load_weekly())
    matchups = _upcoming_matchups(season, week)

    placeholders = _placeholder_player_rows(weekly, position, season, week, matchups)
    if len(placeholders) == 0:
        return pd.DataFrame()

    keep = ["player_id", "player_display_name", "position", "season", "week",
            "team", "opponent_team", config.TARGET] + ROLLING_INPUT_COLS
    history = weekly[weekly["position"] == position][keep].copy()
    history = normalize_team_codes(history, ["team", "opponent_team"])

    # Rolling features + baseline over history WITH the placeholder appended.
    combined = pd.concat([history, placeholders[keep]], ignore_index=True)
    combined = add_rolling_player_features(combined)
    combined = add_rolling_baseline(combined, window=4)

    upcoming = combined[(combined["season"] == season) & (combined["week"] == week)].copy()

    # Opponent defense, with its own placeholder rows so the roll is lagged.
    allowed = build_opponent_defense_features()
    allowed = normalize_team_codes(allowed, ["defense_team"])
    def_placeholders = _placeholder_defense_rows(allowed, season, week, matchups)

    base_cols = ["defense_team", "position", "season", "week", "points_allowed_to_position"]
    def_combined = pd.concat([allowed[base_cols], def_placeholders], ignore_index=True)
    def_combined = def_combined.sort_values(["defense_team", "position", "season", "week"])
    for w in config.ROLLING_WINDOWS:
        def_combined[f"def_points_allowed_roll{w}"] = (
            def_combined.groupby(["defense_team", "position"])["points_allowed_to_position"]
                        .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
    def_upcoming = def_combined[(def_combined["season"] == season) &
                                (def_combined["week"] == week)]

    upcoming = upcoming.merge(
        def_upcoming.drop(columns=["points_allowed_to_position"]),
        left_on=["opponent_team", "position", "season", "week"],
        right_on=["defense_team", "position", "season", "week"],
        how="left",
    )

    # Snap share and expected points, built with this week's rows as placeholders
    # so their rolling values are lagged exactly as in training.
    keys = upcoming[["player_id", "season", "week"]]
    upcoming = upcoming.merge(
        build_snap_features(extra_keys=keys),
        on=["player_id", "season", "week"], how="left",
    )
    upcoming = upcoming.merge(
        build_expected_points_features(extra_keys=keys),
        on=["player_id", "season", "week"], how="left",
    )

    # Schedule/Vegas context, same home/away unpivot as build_features.
    sched = load_schedule_context()
    home = sched.rename(columns={"home_team": "team",
                                 "home_implied_total": "implied_total"})[
        ["season", "week", "team", "implied_total", "spread_line", "total_line"]
    ]
    away = sched.rename(columns={"away_team": "team",
                                 "away_implied_total": "implied_total"})[
        ["season", "week", "team", "implied_total", "spread_line", "total_line"]
    ]
    team_context = pd.concat([home, away], ignore_index=True)

    return upcoming.merge(team_context, on=["season", "week", "team"], how="left")


def _train_through(position, season, week):
    """Fit on completed rows strictly before (season, week) -- same rule as walk-forward."""
    from build_features import build_full_feature_table

    df = build_full_feature_table()
    feature_cols = FEATURE_COLS_BY_POSITION[position]

    mask = (df["position"] == position) & (
        (df["season"] < season) | ((df["season"] == season) & (df["week"] < week))
    )
    train_df = df[mask].dropna(subset=[config.TARGET, "baseline_last4"])

    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
    )
    model.fit(train_df[feature_cols], train_df[config.TARGET])
    return model, len(train_df)


def predict_upcoming(position, season=None, week=None):
    """
    Predictions for the next unplayed week. Returns (DataFrame, meta dict).
    Empty DataFrame if there is no unplayed week in config.SEASONS.

    Columns: player_display_name, team, opponent_team, baseline_pred,
    model_pred, hybrid_pred -- baseline alongside model, per CLAUDE.md.

    `model_pred` is the shipped projection. `hybrid_pred` is retained only for
    continuity with the historical comparison: the hybrid's premise (that the model
    loses to the baseline on low-volume players) stopped being true once the
    expected-points features landed, so its baseline fallback is now pure drag.
    """
    if season is None or week is None:
        found = find_upcoming_week()
        if found is None:
            return pd.DataFrame(), {"season": None, "week": None, "reason": "no unplayed week"}
        season, week = found

    upcoming = build_upcoming_feature_table(position, season, week)
    if len(upcoming) == 0:
        return pd.DataFrame(), {"season": season, "week": week, "reason": "no active players"}

    feature_cols = FEATURE_COLS_BY_POSITION[position]
    threshold = HYBRID_THRESHOLD_BY_POSITION.get(position, DEFAULT_HYBRID_THRESHOLD)

    upcoming = upcoming.dropna(subset=["baseline_last4"])
    if len(upcoming) == 0:
        return pd.DataFrame(), {"season": season, "week": week, "reason": "no baseline history"}

    model, n_train = _train_through(position, season, week)
    upcoming["model_pred"] = model.predict(upcoming[feature_cols])
    upcoming["baseline_pred"] = upcoming["baseline_last4"]
    upcoming["hybrid_pred"] = np.where(
        upcoming["baseline_pred"] >= threshold,
        upcoming["model_pred"],
        upcoming["baseline_pred"],
    )

    out_cols = ["player_display_name", "position", "team", "opponent_team",
                "baseline_pred", "model_pred", "hybrid_pred", "implied_total"]
    result = upcoming[out_cols].sort_values("model_pred", ascending=False).reset_index(drop=True)

    meta = {"season": season, "week": week, "n_train_rows": n_train,
            "n_players": len(result), "hybrid_threshold": threshold}
    return result, meta


if __name__ == "__main__":
    print("upcoming week:", find_upcoming_week())
    for pos in config.ACTIVE_POSITIONS:
        result, meta = predict_upcoming(pos)
        print(f"\n===== {pos} =====")
        print(meta)
        print(result.head(10).to_string(index=False))
