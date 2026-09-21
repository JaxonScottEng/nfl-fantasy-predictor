# pools.py
"""
Which players get scored.

Accuracy depends on the group. Scoring every receiver on a roster includes around
100 bench players a week who score near zero and are easy to predict, which drags
the average error down by about 30%. Published studies score the top 40 per position
per week.

Each pool takes one week of test rows and returns the rows to score.
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
