# data_load.py
import os
import nflreadpy as nfl
import pandas as pd
import config

CACHE_PATH = "data/raw/weekly_stats.parquet"

REGULAR_SEASON = "REG"


def filter_regular_season(df):
    """
    Drop postseason rows.

    Must be applied BEFORE any rolling, or playoff games still enter the windows
    even when they're excluded as prediction targets. Playoff games have a
    different player pool (only 14 teams, and resting starters), so mixing them
    into training and into rolling averages compares unlike things.

    Called explicitly by every module that reads raw weekly data rather than
    hidden inside load_weekly(), because forgetting it in one place is exactly
    the bug this fixes.
    """
    return df[df["season_type"] == REGULAR_SEASON].copy()

def cached_table(name, loader, force_refresh=False):
    """
    Cache any nflverse table to data/raw/<name>.parquet.

    These feed build_full_feature_table(), which the evaluation harness and the
    regression hook both run repeatedly -- without a local cache every run would
    re-download. Converts Polars to pandas once, at this boundary.
    """
    path = f"data/raw/{name}.parquet"

    if os.path.exists(path) and not force_refresh:
        return pd.read_parquet(path)

    print(f"Downloading {name} from nflverse...")
    df = loader()
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    print(f"Cached to {path}")
    return df


def load_weekly(seasons=None, force_refresh=False):
    seasons = seasons or config.SEASONS

    if os.path.exists(CACHE_PATH) and not force_refresh:
        print(f"Loading from cache: {CACHE_PATH}")
        return pd.read_parquet(CACHE_PATH)

    print(f"Downloading seasons {seasons} from nflverse...")
    df = nfl.load_player_stats(seasons)   # Polars DataFrame
    df = df.to_pandas()                   # convert once, at the boundary

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.to_parquet(CACHE_PATH)
    print(f"Cached to {CACHE_PATH}")
    return df

if __name__ == "__main__":
    df = load_weekly()
    print("shape:", df.shape)