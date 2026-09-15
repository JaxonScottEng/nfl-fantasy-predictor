# config.py — single source of truth for tunable values

SEASONS = list(range(2016, 2027))   # 2016–2026; 2026 is IN PROGRESS (partial)
# Seasons after TEST_SEASON are pulled so the GUI can predict the next unplayed
# week. They never reach training/validation: walk_forward_folds trains on
# season < VALIDATION_SEASON and tests on VALIDATION_SEASON only.
ACTIVE_POSITIONS = ["WR", "RB"]   # one separate model per position, never pooled
TARGET = "fantasy_points_ppr"
ROLLING_WINDOWS = [3, 4, 5]          # last-N-game windows for features/baseline

TRAIN_SEASONS = list(range(2016, 2023))   # 2016–2022
VALIDATION_SEASON = 2023
TEST_SEASON = 2024

MAX_VALIDATION_WEEK = 18   # exclude playoff weeks (19-22) from validation