# config.py — single source of truth for tunable values

SEASONS = list(range(2016, 2025))   # 2016–2024, all completed seasons
ACTIVE_POSITIONS = ["WR"]
TARGET = "fantasy_points_ppr"
ROLLING_WINDOWS = [3, 4, 5]          # last-N-game windows for features/baseline