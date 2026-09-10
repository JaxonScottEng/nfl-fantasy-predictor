# build_target.py
import pandas as pd
import config
from data_load import load_weekly

def get_target_table():
    df = load_weekly()
    df = df[df["position"].isin(config.ACTIVE_POSITIONS)].copy()

    # Sort by player then by actual game order — NOT week number.
    # This matters because of bye weeks and season-length differences (16→17 games).
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    keep_cols = ["player_id", "player_display_name", "position",
                "season", "week", "team", "opponent_team",
                config.TARGET]
    # adjust "recent_team" to whatever the actual team column is called —
    # check df.columns if this throws a KeyError
    return df[keep_cols]

if __name__ == "__main__":
    df = get_target_table()
    print("shape:", df.shape)
    print(df.head(10))