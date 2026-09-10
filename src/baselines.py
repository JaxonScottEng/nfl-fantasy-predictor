# baselines.py
import pandas as pd
import config
from build_target import get_target_table

def add_rolling_baseline(df, window=4):
    """
    For each player-week, predict using the MEAN of that player's
    previous `window` games — NOT including the current game.
    This is the critical lag: .shift(1) excludes the row we're predicting.
    """
    df = df.sort_values(["player_id", "season", "week"]).copy()

    df[f"baseline_last{window}"] = (
        df.groupby("player_id")[config.TARGET]
          .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    return df

if __name__ == "__main__":
    df = get_target_table()
    df = add_rolling_baseline(df, window=4)

    # Drop rows with no baseline yet (a player's very first game — nothing to roll from)
    valid = df.dropna(subset=["baseline_last4"])

    mae = (valid[config.TARGET] - valid["baseline_last4"]).abs().mean()
    rmse = ((valid[config.TARGET] - valid["baseline_last4"]) ** 2).mean() ** 0.5

    print(f"Rows evaluated: {len(valid)}")
    print(f"Baseline (last-4-game avg) MAE:  {mae:.3f}")
    print(f"Baseline (last-4-game avg) RMSE: {rmse:.3f}")
    # add temporarily at the bottom of baselines.py, before/after the existing prints
