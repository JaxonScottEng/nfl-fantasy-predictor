"""
End-to-end exercise of the evaluation harness on synthetic data.

run_evaluation() accepts an injected feature_table, so the whole walk-forward loop
-- folds, real model fits, pool selection, metric aggregation -- runs in seconds
instead of the minutes a real season takes.

The property that matters most here is the project's own hard rule: every
comparator must be scored on the IDENTICAL set of rows. Comparing a model against
a baseline measured on a different pool is the exact mistake this harness exists
to prevent.
"""
import pytest

from comparators import DEFAULT_COMPARATOR_IDS
from evaluate import metric_table, run_evaluation, summarize

SEASON = 2021
POOL_SIZE = 5


@pytest.fixture(scope="module")
def evaluated(request):
    table = request.getfixturevalue("synthetic_feature_table")(
        positions=("WR",), seasons=(2020, SEASON), weeks=range(1, 7), n_players=24
    )
    return run_evaluation(
        positions=["WR"], seasons=[SEASON],
        comparator_ids=DEFAULT_COMPARATOR_IDS,
        pool="top_n_by_projection", pool_kwargs={"n": POOL_SIZE},
        feature_table=table, verbose=False,
    )


def test_harness_produces_rows(evaluated):
    assert len(evaluated) > 0
    assert set(evaluated["comparator"]) == set(DEFAULT_COMPARATOR_IDS)


def test_every_comparator_is_scored_on_identical_rows(evaluated):
    """CLAUDE.md's hard rule, enforced mechanically."""
    row_sets = {
        comparator: set(zip(group["player_id"], group["week"]))
        for comparator, group in evaluated.groupby("comparator")
    }
    reference = next(iter(row_sets.values()))
    for comparator, rows in row_sets.items():
        assert rows == reference, f"{comparator} scored a different row set"


def test_pool_selects_exactly_n_players_per_week(evaluated):
    for (comparator, week), group in evaluated.groupby(["comparator", "week"]):
        assert int(group["in_pool"].sum()) == POOL_SIZE, (
            f"{comparator} week {week} selected {int(group['in_pool'].sum())}"
        )


def test_pool_membership_is_identical_across_comparators(evaluated):
    """
    The pool is chosen once from the ranking comparator's projections, not
    per-comparator -- otherwise each one would be judged on its own favourable rows.
    """
    pools = {
        comparator: set(zip(group.loc[group["in_pool"], "player_id"],
                            group.loc[group["in_pool"], "week"]))
        for comparator, group in evaluated.groupby("comparator")
    }
    reference = next(iter(pools.values()))
    assert all(pool == reference for pool in pools.values())


def test_actuals_are_consistent_for_a_given_row(evaluated):
    """The same player-week must carry one actual, whichever comparator scored it."""
    spread = evaluated.groupby(["player_id", "week"])["actual"].nunique()
    assert (spread == 1).all()


def test_summarize_emits_one_row_per_comparator_and_metric(evaluated):
    summary = summarize(evaluated, metric_ids=["mae", "rmse"])

    assert set(summary["comparator"]) == set(DEFAULT_COMPARATOR_IDS)
    assert set(summary["metric"]) == {"mae", "rmse"}
    assert len(summary) == len(DEFAULT_COMPARATOR_IDS) * 2
    assert (summary["n_rows"] > 0).all()
    assert summary["value"].notna().all()


def test_summarize_respects_the_pool_filter(evaluated):
    pooled = summarize(evaluated, metric_ids=["mae"], pool_only=True)
    everything = summarize(evaluated, metric_ids=["mae"], pool_only=False)

    assert (pooled["n_rows"] < everything["n_rows"]).all()


def test_metric_table_pivots_to_comparators_by_metric(evaluated):
    table = metric_table(evaluated, metric_ids=["mae", "rmse"])
    assert "mae" in table.columns
    assert len(table) == len(DEFAULT_COMPARATOR_IDS)
