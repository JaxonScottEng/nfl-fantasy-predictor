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


DEFAULT_COMPARATOR_IDS = ["baseline_last4", "xgb_model", "hybrid"]
