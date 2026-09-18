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

import config
from data_load import load_weekly, filter_regular_season
from features_player import add_rolling_player_features, ROLLING_INPUT_COLS
from features_opponent import build_opponent_defense_features
from features_snap import build_snap_features
from features_expected import build_expected_points_features
from features_context import (
    load_schedule_context,
    load_schedules_cached,
    normalize_team_codes,
)
from baselines import add_rolling_baseline
from train import (
    FEATURE_COLS_BY_POSITION,
    HYBRID_THRESHOLD_BY_POSITION,
    DEFAULT_HYBRID_THRESHOLD,
    XGB_PARAMS,
)

# A player needs at least this many completed games in the current season to be
# treated as active. Filters out last year's roster and long-term absences.
MIN_RECENT_GAMES = 1


def find_upcoming_week(schedule=None):
    """
    (season, week) of the next unplayed game, or None if every configured
    season is complete. "Unplayed" = no result recorded yet.
    """
    sched = schedule if schedule is not None else load_schedules_cached()
    unplayed = sched[sched["home_score"].isna()]
    if len(unplayed) == 0:
        return None

    season = int(unplayed["season"].min())
    week = int(unplayed.loc[unplayed["season"] == season, "week"].min())
    return season, week


def _upcoming_matchups(season, week):
    """One row per team playing that week: team + opponent_team."""
    sched = load_schedules_cached()
    sched = sched[sched["season"] == season]
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

    return upcoming.merge(_team_week_context(), on=["season", "week", "team"], how="left")


def _team_week_context():
    """season, week, team + implied_total/spread_line/total_line, home and away pooled."""
    sched = load_schedule_context()
    cols = ["season", "week", "team", "implied_total", "spread_line", "total_line"]
    home = sched.rename(columns={"home_team": "team",
                                 "home_implied_total": "implied_total"})[cols]
    away = sched.rename(columns={"away_team": "team",
                                 "away_implied_total": "implied_total"})[cols]
    return pd.concat([home, away], ignore_index=True)


def _training_slice(position, season, week):
    """Completed rows strictly before (season, week) -- the walk-forward rule."""
    from build_features import build_full_feature_table

    df = build_full_feature_table()
    mask = (df["position"] == position) & (
        (df["season"] < season) | ((df["season"] == season) & (df["week"] < week))
    )
    return df[mask].dropna(subset=[config.TARGET, "baseline_last4"])


def _train_through(position, season, week):
    """Fit on completed rows strictly before (season, week) -- same rule as walk-forward."""
    feature_cols = FEATURE_COLS_BY_POSITION[position]
    train_df = _training_slice(position, season, week)

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(train_df[feature_cols], train_df[config.TARGET])
    return model, len(train_df)


def _defense_form(season, as_of_week):
    """
    Each defense's lagged points-allowed-to-position as of `as_of_week`, keyed by
    defense team so it can be re-pointed at whatever opponent a later week brings.

    One value per defense serves every future week, for the same reason one model
    fit does: no games are played in between, so the lagged window cannot advance.
    """
    allowed = build_opponent_defense_features()
    allowed = normalize_team_codes(allowed, ["defense_team"])

    teams = sorted(allowed["defense_team"].dropna().unique())
    positions = allowed["position"].dropna().unique()
    placeholders = pd.DataFrame(
        [(t, p, season, as_of_week, np.nan) for t in teams for p in positions],
        columns=["defense_team", "position", "season", "week", "points_allowed_to_position"],
    )

    base_cols = ["defense_team", "position", "season", "week", "points_allowed_to_position"]
    combined = pd.concat([allowed[base_cols], placeholders], ignore_index=True)
    combined = combined.sort_values(["defense_team", "position", "season", "week"])
    for w in config.ROLLING_WINDOWS:
        combined[f"def_points_allowed_roll{w}"] = (
            combined.groupby(["defense_team", "position"])["points_allowed_to_position"]
                    .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )

    out = combined[(combined["season"] == season) & (combined["week"] == as_of_week)]
    return out.drop(columns=["points_allowed_to_position", "season", "week"])


def predict_weeks(position, season, weeks, include_quantiles=True):
    """
    Project several UNPLAYED weeks in one pass. One row per (player, week) with
    model_pred, ensemble_pred, baseline_pred and optionally q10/q50/q90.

    Why a single fit covers all of them: every week here is unplayed, so the
    training data and each player's lagged features are identical across them.
    Only the opponent, that defense's form and the market line change.

    HONEST LIMITATION: beyond the next week this is a CURRENT-FORM OUTLOOK, not a
    game-specific projection. Rolling features cannot advance until the intervening
    games are played, and ~88% of later unplayed games have no spread/total posted,
    so those inputs arrive as NaN. Expect a nearly flat line across future weeks,
    varying only with opponent strength.
    """
    from model_quantile import fit_quantile_model, predict_quantiles
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    weeks = sorted({int(w) for w in weeks})
    if not weeks:
        return pd.DataFrame()

    as_of = weeks[0]
    form = build_upcoming_feature_table(position, season, as_of)
    if len(form) == 0:
        return pd.DataFrame()
    form = form.dropna(subset=["baseline_last4"])
    if len(form) == 0:
        return pd.DataFrame()

    feature_cols = FEATURE_COLS_BY_POSITION[position]
    def_cols = [f"def_points_allowed_roll{w}" for w in config.ROLLING_WINDOWS]
    situational = ["opponent_team", "defense_team", "implied_total",
                   "spread_line", "total_line"] + def_cols
    form_core = form.drop(columns=[c for c in situational if c in form.columns])

    def_form = _defense_form(season, as_of)
    context = _team_week_context()

    # Same three members the shipped ensemble averages during evaluation.
    train_df = _training_slice(position, season, as_of)
    point_model = xgb.XGBRegressor(**XGB_PARAMS)
    point_model.fit(train_df[feature_cols], train_df[config.TARGET])
    qmodel = fit_quantile_model(train_df, feature_cols)
    ridge_model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0, random_state=42),
    )
    ridge_model.fit(train_df[feature_cols], train_df[config.TARGET])

    parts = []
    for week in weeks:
        matchups = _upcoming_matchups(season, week)
        if len(matchups) == 0:
            continue

        rows = form_core.copy()
        rows["week"] = week
        rows = rows.merge(matchups, on="team", how="inner")
        if len(rows) == 0:
            continue

        rows = rows.merge(def_form, left_on=["opponent_team", "position"],
                          right_on=["defense_team", "position"], how="left")
        rows = rows.merge(context, on=["season", "week", "team"], how="left")

        X = rows[feature_cols]
        q = predict_quantiles(qmodel, X)
        rows["model_pred"] = point_model.predict(X)
        rows["q10"], rows["q50"], rows["q90"] = q[:, 0], q[:, 1], q[:, 2]
        rows["ridge_pred"] = ridge_model.predict(X)
        # The shipped ensemble: mean of the two MEAN-optimal models. q50 is produced
        # for the interval only, never averaged in -- see comparators.SHIPPED_COMPARATOR_ID.
        rows["ensemble_pred"] = (rows["model_pred"] + rows["ridge_pred"]) / 2.0
        rows["baseline_pred"] = rows["baseline_last4"]

        keep = ["player_id", "player_display_name", "position", "season", "week",
                "team", "opponent_team", "baseline_pred", "model_pred",
                "ensemble_pred", "q10", "q50", "q90", "implied_total"]
        parts.append(rows[keep])

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def predict_upcoming(position, season=None, week=None):
    """
    Predictions for the next unplayed week. Returns (DataFrame, meta dict).
    Empty DataFrame if there is no unplayed week in config.SEASONS.

    `ensemble_pred` is the shipped projection; `model_pred` and `baseline_pred` sit
    beside it (CLAUDE.md requires the baseline next to every model result), and
    q10/q90 carry the interval. Delegates to predict_weeks so there is exactly one
    definition of how a future week is projected.
    """
    if season is None or week is None:
        found = find_upcoming_week()
        if found is None:
            return pd.DataFrame(), {"season": None, "week": None, "reason": "no unplayed week"}
        season, week = found

    rows = predict_weeks(position, season, [week])
    if len(rows) == 0:
        return pd.DataFrame(), {"season": season, "week": week, "reason": "no active players"}

    out_cols = ["player_display_name", "position", "team", "opponent_team",
                "baseline_pred", "model_pred", "ensemble_pred",
                "q10", "q50", "q90", "implied_total"]
    result = rows[out_cols].sort_values("ensemble_pred", ascending=False).reset_index(drop=True)

    meta = {"season": season, "week": week, "n_players": len(result)}
    return result, meta


if __name__ == "__main__":
    print("upcoming week:", find_upcoming_week())
    for pos in config.ACTIVE_POSITIONS:
        result, meta = predict_upcoming(pos)
        print(f"\n===== {pos} =====")
        print(meta)
        print(result.head(10).to_string(index=False))
