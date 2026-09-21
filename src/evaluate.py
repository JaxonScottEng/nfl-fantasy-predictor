# evaluate.py
"""
The evaluation harness: score any set of comparators, on any player pool, over
any set of seasons, with every comparator judged on identical rows.

Why this exists: the project's headline MAE was measured over every WR with a
prior game (~140 per week), while every published accuracy study measures the top
40 by projection. Those two numbers are not comparable, and the unfiltered one
looks better than it is. See pools.py.

Returns ROW-LEVEL predictions rather than pre-aggregated metrics, because pooled
MAE over all folds is not the mean of per-fold MAEs (folds differ in size) and
week-to-week consistency metrics need the per-week breakdown. summarize() turns
those rows into metrics.

Run from the project root -- the data cache path is relative.
"""
import os

import numpy as np
import pandas as pd

import config
from build_features import build_full_feature_table
from comparators import COMPARATORS, DEFAULT_COMPARATOR_IDS, SHIPPED_COMPARATOR_ID
from metrics import METRICS, DEFAULT_METRIC_IDS
from pools import POOLS, DEFAULT_POOL_SIZE
from validation import walk_forward_folds

REQUIRED_COLS = [config.TARGET, "baseline_last4"]


def run_evaluation(positions=None, seasons=None, comparator_ids=None,
                   pool="all", pool_kwargs=None,
                   pool_ranking_comparator="xgb_model",
                   feature_table=None, verbose=True):
    """
    Walk-forward evaluate comparators and return one row per
    (player-week, comparator) with actual, prediction and pool membership.
    """
    positions = positions or config.ACTIVE_POSITIONS
    seasons = seasons or [config.VALIDATION_SEASON]
    comparator_ids = comparator_ids or DEFAULT_COMPARATOR_IDS
    pool_kwargs = dict(pool_kwargs or {})
    pool_fn = POOLS.get(pool)
    needs_ranking = POOLS.meta(pool, "needs_ranking", False)

    df = feature_table if feature_table is not None else build_full_feature_table()

    parts = []
    for position in positions:
        pos_all = df[df["position"] == position]
        n_default = DEFAULT_POOL_SIZE.get(position)

        for season in seasons:
            # Same postseason trim train.py applies to the validation season.
            pos_df = pos_all[~((pos_all["season"] == season) &
                               (pos_all["week"] > config.MAX_VALIDATION_WEEK))]

            for week, train_df, test_df in walk_forward_folds(pos_df, validation_season=season):
                train_df = train_df.dropna(subset=REQUIRED_COLS)
                test_df = test_df.dropna(subset=REQUIRED_COLS)
                if len(train_df) == 0 or len(test_df) == 0:
                    continue

                cache = {}
                preds = {
                    cid: np.asarray(
                        COMPARATORS.get(cid)(train_df, test_df, position, cache=cache),
                        dtype=float,
                    )
                    for cid in comparator_ids
                }
                actual = test_df[config.TARGET].to_numpy(dtype=float)

                if needs_ranking:
                    rank_pred = preds.get(pool_ranking_comparator)
                    if rank_pred is None:
                        rank_pred = np.asarray(
                            COMPARATORS.get(pool_ranking_comparator)(
                                train_df, test_df, position, cache=cache),
                            dtype=float,
                        )
                    kwargs = dict(pool_kwargs)
                    kwargs.setdefault("n", n_default)
                    ranked = test_df.assign(_pred=rank_pred, _actual=actual)
                    selected = pool_fn(ranked, pred_col="_pred", actual_col="_actual", **kwargs)
                else:
                    selected = pool_fn(test_df, **pool_kwargs)

                in_pool = test_df.index.isin(selected)

                base = pd.DataFrame({
                    "season": season,
                    "week": week,
                    "position": position,
                    "player_id": test_df["player_id"].to_numpy(),
                    "player_display_name": test_df["player_display_name"].to_numpy(),
                    "actual": actual,
                    "in_pool": in_pool,
                })
                for cid, pred in preds.items():
                    part = base.copy()
                    part["comparator"] = cid
                    part["pred"] = pred
                    parts.append(part)

        if verbose:
            print(f"  evaluated {position}: seasons {seasons}")

    if not parts:
        return pd.DataFrame(columns=["season", "week", "position", "player_id",
                                     "player_display_name", "actual", "in_pool",
                                     "comparator", "pred"])
    return pd.concat(parts, ignore_index=True)


def summarize(row_df, metric_ids=None, pool_only=True, by_season=False, **metric_kwargs):
    """Row-level predictions -> long-form metrics per (position, comparator)."""
    metric_ids = metric_ids or DEFAULT_METRIC_IDS
    rows = row_df[row_df["in_pool"]] if pool_only else row_df
    if len(rows) == 0:
        return pd.DataFrame(columns=["position", "comparator", "metric", "value", "n_rows"])

    keys = ["position"] + (["season"] if by_season else []) + ["comparator"]
    out = []
    for key_vals, group in rows.groupby(keys, sort=False):
        key_vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        record_keys = dict(zip(keys, key_vals))
        groups = group["season"].astype(str) + "-w" + group["week"].astype(str)
        y_true = group["actual"].to_numpy(dtype=float)
        y_pred = group["pred"].to_numpy(dtype=float)

        for mid in metric_ids:
            value = METRICS.get(mid)(y_true, y_pred, groups=groups, **metric_kwargs)
            out.append({**record_keys, "metric": mid, "value": value,
                        "n_rows": len(group)})

    return pd.DataFrame(out)


def metric_table(row_df, metric_ids=None, pool_only=True, **metric_kwargs):
    """summarize() pivoted to comparators x metrics, for reading."""
    long_df = summarize(row_df, metric_ids=metric_ids, pool_only=pool_only, **metric_kwargs)
    return long_df.pivot_table(index=["position", "comparator"],
                               columns="metric", values="value", sort=False)


# The Stage 0 acceptance gate: with pool="all" over 2023 the harness must
# reproduce train.py's locked-in numbers exactly, or it is not measuring the
# same thing and no later comparison can be trusted.
FIDELITY_TARGETS = {
    ("WR", "baseline_last4"): 4.685, ("WR", "xgb_model"): 4.384, ("WR", "hybrid"): 4.506,
    ("RB", "baseline_last4"): 4.551, ("RB", "xgb_model"): 4.349, ("RB", "hybrid"): 4.375,
}


# Named explicitly rather than inherited from DEFAULT_COMPARATOR_IDS: this check
# verifies the harness against train.py's three outputs, which include the hybrid.
# When the default set changed to ship the ensemble, the hybrid dropped out of the
# run and this check reported None for numbers train.py still produces.
FIDELITY_COMPARATORS = ["baseline_last4", "xgb_model", "hybrid"]


def check_fidelity(verbose=True):
    rows = run_evaluation(pool="all", seasons=[config.VALIDATION_SEASON],
                          comparator_ids=FIDELITY_COMPARATORS, verbose=False)
    got = summarize(rows, metric_ids=["mae"], pool_only=False)

    failures = []
    for (position, comparator), expected in FIDELITY_TARGETS.items():
        match = got[(got["position"] == position) & (got["comparator"] == comparator)]
        actual = round(float(match["value"].iloc[0]), 3) if len(match) else None
        ok = actual is not None and actual == round(expected, 3)
        if not ok:
            failures.append((position, comparator, expected, actual))
        if verbose:
            flag = "OK  " if ok else "FAIL"
            print(f"  {flag} {position:3s} {comparator:16s} expected {expected:.3f}  got "
                  f"{'None' if actual is None else f'{actual:.3f}'}")

    return failures


DETAIL_PATH = os.path.join("data", "processed", "predictions_detail.csv")
DETAIL_COMPARATORS = ["baseline_last4", "xgb_model", SHIPPED_COMPARATOR_ID,
                      "xgb_q10", "xgb_q50", "xgb_q90"]


def write_prediction_details(seasons=None, path=DETAIL_PATH, feature_table=None):
    """
    Row-level predictions for the GUI, one column per comparator.

    The evaluation layer owns this file rather than train.py, because it is the only
    layer that can reach the comparator registry -- train.py cannot import it without
    a cycle. That also means the shipped ensemble and its quantile interval are
    defined in exactly one place instead of being reimplemented for the GUI.
    """
    rows = run_evaluation(seasons=seasons or [config.VALIDATION_SEASON], pool="all",
                          comparator_ids=DETAIL_COMPARATORS,
                          feature_table=feature_table, verbose=False)
    if len(rows) == 0:
        return rows

    wide = rows.pivot_table(
        index=["season", "week", "position", "player_id", "player_display_name", "actual"],
        columns="comparator", values="pred",
    ).reset_index()
    wide = wide.rename(columns={
        SHIPPED_COMPARATOR_ID: "ensemble_pred",
        "baseline_last4": "baseline_pred",
        "xgb_model": "model_pred",
        "xgb_q10": "q10", "xgb_q50": "q50", "xgb_q90": "q90",
    })

    for name in ("baseline", "model", "ensemble"):
        wide[f"{name}_error"] = (wide["actual"] - wide[f"{name}_pred"]).abs()
    wide["model_advantage"] = wide["baseline_error"] - wide["model_error"]
    wide["ensemble_advantage"] = wide["baseline_error"] - wide["ensemble_error"]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    wide.to_csv(path, index=False)
    return wide


def regression_report():
    """
    Fast single-season top-40 numbers in a machine-readable form, for the
    PostToolUse regression hook. One season only -- the hook fires on every
    src/*.py edit, so the full multi-season run is too slow to belong here.
    """
    rows = run_evaluation(pool="top_n_by_projection",
                          seasons=[config.VALIDATION_SEASON], verbose=False)
    long_df = summarize(rows, metric_ids=["mae", "rmse"])
    for _, r in long_df.iterrows():
        print(f"REGRESSION {r['position']} {r['comparator']} "
              f"{r['metric'].upper()}: {r['value']:.3f}")
    return long_df


if __name__ == "__main__":
    import sys

    if "--regression" in sys.argv:
        regression_report()
        raise SystemExit(0)

    print("Harness fidelity check (pool=all, 2023) -- must match train.py exactly:")
    failures = check_fidelity()
    print()
    if failures:
        raise SystemExit(f"FIDELITY CHECK FAILED for {len(failures)} comparator(s) -- "
                         "the harness is not measuring what train.py measures.")
    print("Fidelity check PASSED.\n")

    seasons = [int(a) for a in sys.argv[1:] if a.isdigit()] or [config.VALIDATION_SEASON]
    print(f"Comparable pool (top 40 WR/RB by projection), seasons {seasons}:")
    rows = run_evaluation(pool="top_n_by_projection", seasons=seasons,
                          comparator_ids=DETAIL_COMPARATORS, verbose=False)
    point_rows = rows[~rows["comparator"].isin(["xgb_q10", "xgb_q90"])]
    print(metric_table(point_rows, n=12).round(3).to_string())
    print("\nPublished reference (top-40 pool): WR 4.84-4.94 MAE, RB 5.06-5.20.")

    # Interval calibration, so the coverage figures quoted in CLAUDE.md and REPORT.md
    # are reproducible by running something rather than taken on trust.
    from model_quantile import interval_coverage, pinball_loss

    wide = rows[rows["comparator"].isin(["xgb_q10", "xgb_q50", "xgb_q90"])].pivot_table(
        index=["season", "week", "position", "player_id", "actual", "in_pool"],
        columns="comparator", values="pred",
    ).reset_index()
    print("\nInterval calibration (q10-q90, target coverage 0.80):")
    for position in sorted(wide["position"].unique()):
        pooled = wide[(wide["position"] == position) & wide["in_pool"]]
        coverage = interval_coverage(pooled["actual"], pooled["xgb_q10"], pooled["xgb_q90"])
        width = float((pooled["xgb_q90"] - pooled["xgb_q10"]).mean())
        pinball = pinball_loss(pooled["actual"], pooled["xgb_q50"], 0.5)
        print(f"  {position}: coverage {coverage:.3f}  mean width {width:5.2f}  "
              f"pinball(0.5) {pinball:.3f}  n={len(pooled):,}")
