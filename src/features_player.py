# features_player.py
"""
Player usage/efficiency features — all rolling, all lagged.
Lag is applied the same way as the baseline: shift(1) before
rolling, grouped by player_id, so the current week is never included.
"""
import numpy as np
import config
from data_load import load_weekly, filter_regular_season

# Raw stat columns we'll build rolling features from.
# Verified against the real column list. Adjust here if
# nflreadpy's schema changes.
USAGE_COLS = ["targets", "receptions", "receiving_air_yards", "carries"]
PROD_COLS = ["receiving_yards", "rushing_yards", "receiving_tds", "rushing_tds",
             "rushing_first_downs"]

# Team-context-normalized opportunity and efficiency. Raw target counts can't say
# whether 6 targets was a big share of a low-volume offense or a small share of a
# pass-heavy one; target_share and air_yards_share can. All are single-game values
# (verified: target_share sums to 1.0 per team-week), so they are only safe because
# they go through the same shift-then-roll treatment as everything else.
# Deliberately excluded: `wopr` (exactly 1.5*target_share + 0.7*air_yards_share, so
# the primitives carry it), `racr` (explodes as air yards approach zero) and `pacr`
# (99%+ null off QB).
OPPORTUNITY_COLS = ["target_share", "air_yards_share", "receiving_epa", "rushing_epa",
                    "receiving_first_downs", "receiving_yards_after_catch"]

# Every raw column that gets rolled. Single list so callers that rebuild this
# pipeline (upcoming.py) cannot fall out of sync with it.
ROLLING_INPUT_COLS = USAGE_COLS + PROD_COLS + OPPORTUNITY_COLS

def add_rolling_player_features(df, windows=None):
    windows = windows or config.ROLLING_WINDOWS
    df = df.sort_values(["player_id", "season", "week"]).copy()

    for col in ROLLING_INPUT_COLS:
        for w in windows:
            new_col = f"{col}_roll{w}"
            df[new_col] = (
                df.groupby("player_id")[col]
                  .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )

    for w in windows:
        df[f"yards_per_target_roll{w}"] = (
            df[f"receiving_yards_roll{w}"] / df[f"targets_roll{w}"].replace(0, np.nan)
        ).astype(float)

        # RB-oriented. Carries are the dominant RB usage signal (RB mean 7.7
        # carries/game vs WR 0.18), so RBs need a rushing-side efficiency ratio
        # and a combined-opportunity count the way WRs need yards-per-target.
        # Built from the already-lagged rolling columns, same as above.
        df[f"yards_per_carry_roll{w}"] = (
            df[f"rushing_yards_roll{w}"] / df[f"carries_roll{w}"].replace(0, np.nan)
        ).astype(float)

        df[f"touches_roll{w}"] = (
            df[f"carries_roll{w}"] + df[f"receptions_roll{w}"]
        ).astype(float)

    return df

def build_player_features():
    raw = filter_regular_season(load_weekly())
    wr = raw[raw["position"].isin(config.ACTIVE_POSITIONS)].copy()
    keep = ["player_id", "player_display_name", "position", "season", "week",
            "team", "opponent_team", config.TARGET] + ROLLING_INPUT_COLS
    wr = wr[keep]
    return add_rolling_player_features(wr)

if __name__ == "__main__":
    df = build_player_features()
    print("shape:", df.shape)
    print(df.columns.tolist())
    print(df[["yards_per_target_roll3", "yards_per_target_roll4", "yards_per_target_roll5"]].dtypes)
    print(df[df["player_display_name"] == "Steve Smith"].head(8))