"""Shared fixtures. The fast tier uses synthetic frames only -- no network, no cache."""
import os

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session", autouse=True)
def project_root():
    """
    The pipeline resolves its cache paths relative to the working directory, so a
    test run started from elsewhere would read a different cache. Pin it once.
    """
    previous = os.getcwd()
    os.chdir(PROJECT_ROOT)
    yield PROJECT_ROOT
    os.chdir(previous)


@pytest.fixture
def tiny_weekly():
    """
    Factory for a minimal weekly-stats frame carrying every column the rolling
    feature builder consumes. Values are deterministic so assertions can be
    hand-computed rather than approximated.
    """
    from features_player import ROLLING_INPUT_COLS

    def _make(players=("00-0000001", "00-0000002"), season=2020, weeks=range(1, 7),
              season_type="REG", position="WR", start=1.0, step=1.0):
        rows = []
        for p_index, player in enumerate(players):
            for w in weeks:
                row = {
                    "player_id": player,
                    "player_display_name": f"Player {p_index + 1}",
                    "position": position,
                    "season": season,
                    "week": w,
                    "season_type": season_type,
                    "team": "SF",
                    "opponent_team": "LA",
                    "fantasy_points_ppr": float(w),
                }
                # Player 1 gets 1,2,3...; player 2 gets 100,200,300... so any
                # cross-player contamination is glaring rather than subtle.
                scale = 1.0 if p_index == 0 else 100.0
                for col in ROLLING_INPUT_COLS:
                    row[col] = (start + (w - 1) * step) * scale
                rows.append(row)
        return pd.DataFrame(rows)

    return _make


@pytest.fixture(scope="session")
def synthetic_feature_table():
    """
    A feature table shaped like build_full_feature_table()'s output: every column
    the position's model expects, plus identity/target columns. Random but seeded.
    """
    from train import FEATURE_COLS_BY_POSITION

    def _make(positions=("WR",), seasons=(2020, 2021), weeks=range(1, 7), n_players=24):
        rng = np.random.default_rng(0)
        frames = []
        for position in positions:
            feature_cols = FEATURE_COLS_BY_POSITION[position]
            for season in seasons:
                for week in weeks:
                    block = pd.DataFrame({
                        "player_id": [f"00-{position}{i:05d}" for i in range(n_players)],
                        "player_display_name": [f"{position} {i}" for i in range(n_players)],
                        "position": position,
                        "season": season,
                        "week": week,
                    })
                    for col in feature_cols:
                        block[col] = rng.normal(8.0, 4.0, n_players)
                    # Target correlates with baseline_last4 so the models have real
                    # signal to find; pure noise would make fits degenerate.
                    block["fantasy_points_ppr"] = (
                        block["baseline_last4"] + rng.normal(0.0, 2.0, n_players)
                    )
                    frames.append(block)
        return pd.concat(frames, ignore_index=True)

    return _make


@pytest.fixture
def skip_if_no_cache():
    """Slow-tier guard: the nflverse cache is gitignored, so it may not exist."""
    cache = os.path.join(PROJECT_ROOT, "data", "raw", "weekly_stats.parquet")
    if not os.path.exists(cache):
        pytest.skip("nflverse cache absent; run `python src/data_load.py` first")
    return cache
