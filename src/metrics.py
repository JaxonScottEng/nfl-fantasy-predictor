# metrics.py
"""
Accuracy metrics, as a registry.

Every metric takes (y_true, y_pred) and optional `groups` -- a per-row label
identifying the (season, week) an observation belongs to. Metrics that judge
ordering or week-to-week consistency need `groups`; scale metrics ignore it.

The metric set mirrors what the fantasy analytics community actually reports:
MAE (primary), RMSE, R^2, mean error (directional bias), Spearman within
position, top-N hit rate, and the coefficient of variation of weekly MAE
(week-to-week consistency).
"""
import numpy as np
import pandas as pd

from registry import Registry

METRICS = Registry("metric")


def _as_groups(groups, metric_id):
    if groups is None:
        raise ValueError(
            f"metric {metric_id!r} needs `groups` (per-row season-week labels)"
        )
    return pd.Series(np.asarray(groups)).reset_index(drop=True)


@METRICS.register("mae", label="MAE")
def mae(y_true, y_pred, **_):
    return float(np.mean(np.abs(y_true - y_pred)))


@METRICS.register("rmse", label="RMSE")
def rmse(y_true, y_pred, **_):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


@METRICS.register("r2", label="R^2")
def r2(y_true, y_pred, **_):
    y_true = np.asarray(y_true, dtype=float)
    ss_res = np.sum((y_true - np.asarray(y_pred, dtype=float)) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


@METRICS.register("mean_error", label="Mean error (bias)",
                  caveat="Negative = under-projecting on average.")
def mean_error(y_true, y_pred, **_):
    return float(np.mean(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))


@METRICS.register("spearman_within", label="Spearman (within week)",
                  caveat="Averaged over weeks. Never pool positions -- QB's scale would dominate.")
def spearman_within(y_true, y_pred, *, groups=None, **_):
    from scipy.stats import spearmanr

    g = _as_groups(groups, "spearman_within")
    y_true = pd.Series(np.asarray(y_true, dtype=float)).reset_index(drop=True)
    y_pred = pd.Series(np.asarray(y_pred, dtype=float)).reset_index(drop=True)

    per_week = []
    for _, idx in g.groupby(g, sort=False).groups.items():
        t, p = y_true.loc[idx], y_pred.loc[idx]
        if len(t) < 3 or t.nunique() < 2 or p.nunique() < 2:
            continue
        rho = spearmanr(t, p).statistic
        if not np.isnan(rho):
            per_week.append(rho)

    return float(np.mean(per_week)) if per_week else float("nan")


@METRICS.register("top_n_hit_rate", label="Top-N hit rate",
                  caveat="Share of the predicted top N that were actually in the top N that week.")
def top_n_hit_rate(y_true, y_pred, *, groups=None, n=12, **_):
    g = _as_groups(groups, "top_n_hit_rate")
    y_true = pd.Series(np.asarray(y_true, dtype=float)).reset_index(drop=True)
    y_pred = pd.Series(np.asarray(y_pred, dtype=float)).reset_index(drop=True)

    rates = []
    for _, idx in g.groupby(g, sort=False).groups.items():
        t, p = y_true.loc[idx], y_pred.loc[idx]
        k = min(n, len(t))
        if k == 0:
            continue
        hit = len(set(p.nlargest(k).index) & set(t.nlargest(k).index))
        rates.append(hit / k)

    return float(np.mean(rates)) if rates else float("nan")


@METRICS.register("cov_weekly_mae", label="CoV of weekly MAE",
                  caveat="std/mean of per-week MAE -- lower means more consistent week to week.")
def cov_weekly_mae(y_true, y_pred, *, groups=None, **_):
    g = _as_groups(groups, "cov_weekly_mae")
    err = pd.Series(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)))
    weekly = err.groupby(g).mean()
    if len(weekly) < 2 or weekly.mean() == 0:
        return float("nan")
    return float(weekly.std(ddof=1) / weekly.mean())


DEFAULT_METRIC_IDS = ["mae", "rmse", "r2", "mean_error",
                      "spearman_within", "top_n_hit_rate", "cov_weekly_mae"]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.gamma(2, 4, 200)
    p = y + rng.normal(0, 3, 200)
    weeks = pd.Series(np.repeat(np.arange(10), 20)).astype(str)
    for mid in DEFAULT_METRIC_IDS:
        print(f"{METRICS.label(mid):26s} {METRICS.get(mid)(y, p, groups=weeks, n=5):.4f}")
