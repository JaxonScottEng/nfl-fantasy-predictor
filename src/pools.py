# pools.py
"""
Player-pool definitions -- which rows of a week get scored.

This is the whole comparability story. Published accuracy studies do NOT
evaluate every player: Fantasy Football Analytics uses the top 40 WR/RB and top
20 QB/TE *by projected points*, and FantasyPros uses the union of top-N-by-rank
and top-N-by-actual. Scoring all ~140 WRs in a week instead makes MAE look far
better than it is, because deep-bench players score near zero and are trivial to
predict. A number produced over one pool cannot be compared to a number produced
over another.

Each pool takes one week's test frame and returns the index of rows to score.
"""
from registry import Registry

POOLS = Registry("pool")


@POOLS.register("all", label="All rows",
                caveat="Not comparable to published figures -- deep-bench players flatter MAE.",
                needs_ranking=False)
def all_rows(test_df, **_):
    return test_df.index


@POOLS.register("top_n_by_projection", label="Top N by projection",
                caveat="Matches Fantasy Football Analytics' methodology (top 40 WR/RB, top 20 QB/TE).",
                needs_ranking=True)
def top_n_by_projection(test_df, *, pred_col, n, **_):
    return test_df.nlargest(n, pred_col).index


@POOLS.register("union_top_n", label="Top N by projection or actual",
                caveat="Matches FantasyPros' pool: top-N-by-rank union top-N-by-actual.",
                needs_ranking=True)
def union_top_n(test_df, *, pred_col, actual_col, n, **_):
    return test_df.nlargest(n, pred_col).index.union(
        test_df.nlargest(n, actual_col).index
    )


# Published study convention, for reference when choosing `n`.
DEFAULT_POOL_SIZE = {"WR": 40, "RB": 40, "QB": 20, "TE": 20}
