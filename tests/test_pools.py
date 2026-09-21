"""
Player-pool selection -- the mechanism the whole comparability claim rests on.

Scoring every WR with a prior game (~140/week) instead of the top 40 flattered MAE
by roughly 30%. These tests keep the pool definitions honest and, crucially, keep
every pool carrying the caveat that explains what it is and is not comparable to.
"""
import pandas as pd
import pytest

from pools import POOLS, DEFAULT_POOL_SIZE


@pytest.fixture
def week_frame():
    return pd.DataFrame({
        "player_id": [f"p{i}" for i in range(10)],
        "_pred": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        # Deliberately anti-correlated so top-by-projection and top-by-actual differ.
        "_actual": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    })


def test_all_rows_returns_everything(week_frame):
    selected = POOLS.get("all")(week_frame)
    assert len(selected) == len(week_frame)


def test_top_n_by_projection_takes_the_n_largest(week_frame):
    selected = POOLS.get("top_n_by_projection")(week_frame, pred_col="_pred", n=3)
    assert set(week_frame.loc[selected, "_pred"]) == {10, 9, 8}


def test_top_n_handles_n_larger_than_the_frame(week_frame):
    selected = POOLS.get("top_n_by_projection")(week_frame, pred_col="_pred", n=99)
    assert len(selected) == len(week_frame)


def test_union_top_n_is_a_superset_of_both_halves(week_frame):
    n = 3
    union = POOLS.get("union_top_n")(week_frame, pred_col="_pred", actual_col="_actual", n=n)
    by_pred = POOLS.get("top_n_by_projection")(week_frame, pred_col="_pred", n=n)

    assert set(by_pred).issubset(set(union))
    assert n <= len(union) <= 2 * n


def test_every_pool_declares_a_caveat():
    """
    A pool without a stated caveat is how a number quietly becomes incomparable
    again. Registering one without an explanation should fail here.
    """
    for pool_id in POOLS.available():
        caveat = POOLS.caveat(pool_id)
        assert caveat and caveat.strip(), f"pool {pool_id!r} has no caveat"


def test_published_study_pool_sizes_are_declared():
    """Matches the convention published accuracy studies use."""
    assert DEFAULT_POOL_SIZE["WR"] == 40
    assert DEFAULT_POOL_SIZE["RB"] == 40
    assert DEFAULT_POOL_SIZE["QB"] == 20
    assert DEFAULT_POOL_SIZE["TE"] == 20
