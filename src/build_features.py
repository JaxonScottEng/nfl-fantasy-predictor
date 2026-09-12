# build_features.py
import pandas as pd
from features_player import build_player_features
from features_opponent import build_opponent_defense_features
from features_context import load_schedule_context, normalize_team_codes

def build_full_feature_table():
    player = build_player_features()
    defense = build_opponent_defense_features()
    sched = load_schedule_context()

    # Normalize the player table's team codes too, so both sides of
    # every join use the same convention (current codes: LV, LAC, etc.)
    player = normalize_team_codes(player, ["team", "opponent_team"])
    defense = normalize_team_codes(defense, ["defense_team"])

    df = player.merge(
        defense,
        left_on=["opponent_team", "season", "week"],
        right_on=["defense_team", "season", "week"],
        how="left"
    )

    # Join schedule context -- need home/away perspective per player's team.
    home = sched.rename(columns={"home_team": "team",
                                  "home_implied_total": "implied_total",
                                  "away_team": "opp_temp"})[
        ["season", "week", "team", "implied_total", "spread_line", "total_line"]
    ]
    away = sched.rename(columns={"away_team": "team",
                                  "away_implied_total": "implied_total",
                                  "home_team": "opp_temp"})[
        ["season", "week", "team", "implied_total", "spread_line", "total_line"]
    ]
    team_context = pd.concat([home, away], ignore_index=True)

    df = df.merge(team_context, on=["season", "week", "team"], how="left")

    return df

if __name__ == "__main__":
    df = build_full_feature_table()
    print("shape:", df.shape)
    print("null implied_total rows:", df["implied_total"].isna().sum())
    print("null def_points_allowed_roll4 rows:", df["def_points_allowed_roll4"].isna().sum())

    null_implied = df[df["implied_total"].isna()]
    print("remaining implied_total null teams:", null_implied["team"].unique() if len(null_implied) else "none")