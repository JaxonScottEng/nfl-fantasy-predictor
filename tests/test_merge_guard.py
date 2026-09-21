"""
The join guard in build_features.

A duplicated key on the right-hand side of a left join silently multiplies rows.
Nothing raises, every downstream metric is computed over inflated data, and the
numbers look plausible. This assertion is the only thing standing between that bug
and a published result.
"""
import pandas as pd
import pytest

from build_features import _merge_player_week


@pytest.fixture
def left():
    return pd.DataFrame({
        "player_id": ["a", "b", "c"],
        "season": [2021, 2021, 2021],
        "week": [1, 1, 1],
        "value": [1.0, 2.0, 3.0],
    })


def test_clean_join_preserves_rows_and_order(left):
    right = pd.DataFrame({
        "player_id": ["a", "b", "c"],
        "season": [2021, 2021, 2021],
        "week": [1, 1, 1],
        "extra": [10.0, 20.0, 30.0],
    })

    out = _merge_player_week(left, right, "extra table")

    assert len(out) == len(left)
    assert list(out["player_id"]) == ["a", "b", "c"]
    assert list(out["extra"]) == [10.0, 20.0, 30.0]


def test_duplicate_keys_raise_and_name_the_join(left):
    right = pd.DataFrame({
        "player_id": ["a", "a"],           # duplicated key
        "season": [2021, 2021],
        "week": [1, 1],
        "extra": [10.0, 11.0],
    })

    with pytest.raises(ValueError, match="snap share"):
        _merge_player_week(left, right, "snap share")


def test_missing_keys_become_nan_rather_than_dropping_rows(left):
    right = pd.DataFrame({
        "player_id": ["a"],
        "season": [2021],
        "week": [1],
        "extra": [10.0],
    })

    out = _merge_player_week(left, right, "partial table")

    assert len(out) == len(left)
    assert out["extra"].isna().sum() == 2
