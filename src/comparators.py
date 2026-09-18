# comparators.py
"""
Things we can compare against each other, as a registry.

A comparator turns one walk-forward fold into predictions for that fold's test
rows. Every comparator is scored on the IDENTICAL row set, which is how
CLAUDE.md's "baseline reported next to every model result, same rows" rule is
enforced at the evaluation layer.

Adding a competitor = write a function, decorate it. No caller edits.

`cache` is a per-fold dict supplied by the harness. The hybrid needs the model's
predictions, so without it evaluating both would fit XGBoost twice per fold.
"""
import numpy as np
import xgboost as xgb

import config
from features_expected import EXPECTED_BASELINE_COL
from model_quantile import QUANTILE_ALPHAS, fit_quantile_model, predict_quantiles
from registry import Registry
from train import (
    FEATURE_COLS_BY_POSITION,
    HYBRID_THRESHOLD_BY_POSITION,
    DEFAULT_HYBRID_THRESHOLD,
    XGB_PARAMS,
)

COMPARATORS = Registry("comparator")


@COMPARATORS.register("baseline_last4", label="Baseline (last-4 avg)", provides={"points"})
def baseline_last4(train_df, test_df, position, **_):
    return test_df["baseline_last4"].to_numpy(dtype=float)


@COMPARATORS.register("xgb_model", label="Model (XGBoost)", provides={"points"})
def xgb_model(train_df, test_df, position, *, cache=None, **_):
    key = ("xgb_model", position)
    if cache is not None and key in cache:
        return cache[key]

    feature_cols = FEATURE_COLS_BY_POSITION[position]
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(train_df[feature_cols], train_df[config.TARGET])
    preds = np.asarray(model.predict(test_df[feature_cols]), dtype=float)

    if cache is not None:
        cache[key] = preds
    return preds


@COMPARATORS.register("hybrid", label="Hybrid (segmented)", provides={"points"})
def hybrid(train_df, test_df, position, *, cache=None, **_):
    preds = xgb_model(train_df, test_df, position, cache=cache)
    baseline = test_df["baseline_last4"].to_numpy(dtype=float)
    threshold = HYBRID_THRESHOLD_BY_POSITION.get(position, DEFAULT_HYBRID_THRESHOLD)
    return np.where(baseline >= threshold, preds, baseline)


@COMPARATORS.register("exp_points_last4", label="Expected points (last-4 avg)",
                      caveat="nflverse expected points, lagged. Falls back to the "
                             "actual-points baseline where unavailable (~14% of rows).",
                      provides={"points"})
def exp_points_last4(train_df, test_df, position, **_):
    """
    Project a player at his recent EXPECTED points rather than his recent actual
    points. Same shape as the naive baseline, but measures what his opportunities
    were worth -- which is far less touchdown-luck-driven than what he scored.
    """
    preds = test_df[EXPECTED_BASELINE_COL].to_numpy(dtype=float)
    fallback = test_df["baseline_last4"].to_numpy(dtype=float)
    return np.where(np.isnan(preds), fallback, preds)


def _quantile_predictions(train_df, test_df, position, cache):
    """One multi-quantile fit per fold, shared by the q10/q50/q90 comparators."""
    key = ("xgb_quantile", position)
    if cache is not None and key in cache:
        return cache[key]

    feature_cols = FEATURE_COLS_BY_POSITION[position]
    model = fit_quantile_model(train_df, feature_cols, alphas=QUANTILE_ALPHAS)
    preds = predict_quantiles(model, test_df[feature_cols])

    if cache is not None:
        cache[key] = preds
    return preds


@COMPARATORS.register("xgb_q50", label="Model (quantile median)", provides={"points"},
                      caveat="Trained on reg:quantileerror at alpha=0.5, which optimizes "
                             "MAE directly rather than squared error.")
def xgb_q50(train_df, test_df, position, *, cache=None, **_):
    return _quantile_predictions(train_df, test_df, position, cache)[:, 1]


@COMPARATORS.register("xgb_q10", label="Model (10th percentile)", provides={"points"},
                      caveat="A floor, not a projection -- do not read its MAE as accuracy.")
def xgb_q10(train_df, test_df, position, *, cache=None, **_):
    return _quantile_predictions(train_df, test_df, position, cache)[:, 0]


@COMPARATORS.register("xgb_q90", label="Model (90th percentile)", provides={"points"},
                      caveat="A ceiling, not a projection -- do not read its MAE as accuracy.")
def xgb_q90(train_df, test_df, position, *, cache=None, **_):
    return _quantile_predictions(train_df, test_df, position, cache)[:, 2]


@COMPARATORS.register("ridge", label="Ridge regression", provides={"points"},
                      caveat="Median-imputed and standardized; included for model-class "
                             "diversity in the ensemble, not as a standalone contender.")
def ridge(train_df, test_df, position, *, cache=None, **_):
    """
    A linear model over the same features. Weak alone, but it errs differently from
    a tree ensemble, which is the only reason an average of the two can beat either.
    """
    key = ("ridge", position)
    if cache is not None and key in cache:
        return cache[key]

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feature_cols = FEATURE_COLS_BY_POSITION[position]
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=1.0, random_state=42),
    )
    pipeline.fit(train_df[feature_cols], train_df[config.TARGET])
    preds = np.asarray(pipeline.predict(test_df[feature_cols]), dtype=float)

    if cache is not None:
        cache[key] = preds
    return preds


def register_ensemble(id, member_ids, weights=None, label=None, caveat=None):
    """
    Register an equal-weight (or fixed-weight) average of other comparators.

    Aggregation is the most reproducible finding in the projection-accuracy
    literature -- consensus beats its own members across positions and eras. Members
    are resolved through the fold cache, so an ensemble containing xgb_model costs no
    extra fit when xgb_model is also being scored.
    """
    @COMPARATORS.register(id, label=label or f"Ensemble ({', '.join(member_ids)})",
                          caveat=caveat, provides={"points"}, members=member_ids)
    def _ensemble(train_df, test_df, position, *, cache=None, **_):
        stacked = np.vstack([
            np.asarray(COMPARATORS.get(m)(train_df, test_df, position, cache=cache),
                       dtype=float)
            for m in member_ids
        ])
        return np.average(stacked, axis=0, weights=weights)

    return _ensemble


register_ensemble(
    "ensemble_xgb_ridge",
    ["xgb_model", "ridge"],
    label="Ensemble (XGBoost + ridge)",
    caveat="Averages only the MEAN-optimal models. A median model was measured and "
           "deliberately excluded -- see below.",
)

# THE SHIPPED PROJECTION. Everything user-facing reads this one id.
#
# Averaging only the mean-optimal models is a deliberate choice against the raw MAE
# leader. Adding xgb_q50 measured BETTER on MAE (WR 6.427 vs 6.474, RB 5.973 vs
# 6.029) but its bias is structural, not tunable: a median sits below the mean on a
# right-skewed distribution, and skew grows with volume, so the error concentrates on
# exactly the players that matter. Measured -0.84 on average but about -4.4 on an
# elite WR, which would read visibly wrong beside any commercial projection. The
# 3-way and 4-way variants are in git history at the Phase 12 commit; re-adding one
# is a single register_ensemble() call.
SHIPPED_COMPARATOR_ID = "ensemble_xgb_ridge"

DEFAULT_COMPARATOR_IDS = ["baseline_last4", "xgb_model", SHIPPED_COMPARATOR_ID]

# Everything worth scoring in a full comparison run.
ALL_POINT_COMPARATOR_IDS = [
    "baseline_last4", "exp_points_last4", "hybrid",
    "ridge", "xgb_model", "xgb_q50", SHIPPED_COMPARATOR_ID,
]

# Quantile columns that ride along with the shipped projection to form its interval.
QUANTILE_COMPARATOR_IDS = ["xgb_q10", "xgb_q50", "xgb_q90"]
