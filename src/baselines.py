# baselines.py
import config

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

