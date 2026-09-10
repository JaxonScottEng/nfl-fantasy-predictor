# data_load.py
import os
import nflreadpy as nfl
import pandas as pd
import config

CACHE_PATH = "data/raw/weekly_stats.parquet"

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