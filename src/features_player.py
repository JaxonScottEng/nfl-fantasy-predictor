# features_player.py
"""
Player usage/efficiency features — all rolling, all lagged.
Lag is applied the same way as the Phase 1 baseline: shift(1) before
rolling, grouped by player_id, so the current week is never included.
"""
import pandas as pd
import config
from data_load import load_weekly

# Raw stat columns we'll build rolling features from.
# Verified against the real column list from Phase 0 — adjust here if
# nflreadpy's schema changes.
USAGE_COLS = ["targets", "receptions", "receiving_air_yards", "carries"]
PROD_COLS = ["receiving_yards", "rushing_yards", "receiving_tds", "rushing_tds"]

def add_rolling_player_features(df, windows=None):
    windows = windows or config.ROLLING_WINDOWS
    df = df.sort_values(["player_id", "season", "week"]).copy()

    for col in USAGE_COLS + PROD_COLS:
        for w in windows:
            new_col = f"{col}_roll{w}"
            df[new_col] = (
                df.groupby("player_id")[col]
                  .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )

    # Efficiency ratios computed FROM the rolled (lagged) values,
    # not from raw same-week stats -- otherwise this reintroduces leakage.
    for w in windows:
        df[f"yards_per_target_roll{w}"] = (
            df[f"receiving_yards_roll{w}"] / df[f"targets_roll{w}"].replace(0, pd.NA)
        )

    return df

def build_player_features():
    raw = load_weekly()
    wr = raw[raw["position"].isin(config.ACTIVE_POSITIONS)].copy()
    keep = ["player_id", "player_display_name", "position", "season", "week",
            "team", "opponent_team", config.TARGET] + USAGE_COLS + PROD_COLS
    wr = wr[keep]
    return add_rolling_player_features(wr)

if __name__ == "__main__":
    df = build_player_features()
    print("shape:", df.shape)
    print(df.columns.tolist())
    print(df[df["player_display_name"] == "Steve Smith"].head(8))