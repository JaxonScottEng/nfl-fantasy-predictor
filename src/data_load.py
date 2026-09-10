# data_load.py
import nflreadpy as nfl

def load_weekly(seasons):
    df = nfl.load_player_stats(seasons)   # nflreadpy returns a POLARS DataFrame
    return df.to_pandas()                 # convert once, at the boundary

if __name__ == "__main__":
    df = load_weekly([2024])
    print("shape:", df.shape)
    print("columns:", df.columns.tolist())
    print(df.head())