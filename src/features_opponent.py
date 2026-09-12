# features_opponent.py
"""
Opponent defensive strength: fantasy points allowed to the position,
rolling and lagged -- computed the same careful way as player features.
"""
import config
from data_load import load_weekly

def build_opponent_defense_features(windows=None):
    windows = windows or config.ROLLING_WINDOWS
    raw = load_weekly()
    pos_df = raw[raw["position"].isin(config.ACTIVE_POSITIONS)].copy()

    # Points allowed BY a team TO the position, per week they played defense.
    # Group by the team that was ON DEFENSE that week (opponent_team here,
    # from the offensive player's row), summing fantasy points allowed.
    allowed = (
        pos_df.groupby(["opponent_team", "season", "week"])[config.TARGET]
              .sum()
              .reset_index()
              .rename(columns={"opponent_team": "defense_team",
                                config.TARGET: "points_allowed_to_position"})
    )
    allowed = allowed.sort_values(["defense_team", "season", "week"])

    for w in windows:
        allowed[f"def_points_allowed_roll{w}"] = (
            allowed.groupby("defense_team")["points_allowed_to_position"]
                   .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )

    return allowed

if __name__ == "__main__":
    df = build_opponent_defense_features()
    print("shape:", df.shape)
    print(df.head(10))