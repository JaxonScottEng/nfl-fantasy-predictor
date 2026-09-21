# model_quantile.py
"""
Quantile regression: predict a range instead of a single number.

Two uses. Training at alpha=0.5 optimizes MAE directly, because MAE is minimized by
the median while the default objective targets the mean. And q10/q90 give a range
learned per player, so it widens for unpredictable players.

XGBoost fits all three quantiles in one model and returns them already ordered.
"""
import numpy as np
import xgboost as xgb

import config

QUANTILE_ALPHAS = (0.1, 0.5, 0.9)
QUANTILE_COLS = ("q10", "q50", "q90")
MEDIAN_INDEX = 1


def fit_quantile_model(train_df, feature_cols, alphas=QUANTILE_ALPHAS, params=None):
    """Same hyperparameters as the point model, so the comparison is apples to apples."""
    from train import XGB_PARAMS

    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=list(alphas),
        **(params or XGB_PARAMS),
    )
    model.fit(train_df[feature_cols], train_df[config.TARGET])
    return model


def predict_quantiles(model, X):
    """(n_rows, n_alphas) array of quantile predictions."""
    preds = np.asarray(model.predict(X), dtype=float)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    return preds


def pinball_loss(y_true, y_pred, alpha):
    """
    The loss quantile regression actually minimizes -- the honest way to score a
    quantile prediction, since MAE only makes sense for the median.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    delta = y_true - y_pred
    return float(np.mean(np.maximum(alpha * delta, (alpha - 1.0) * delta)))


def interval_coverage(y_true, lo, hi):
    """
    Share of actuals landing inside [lo, hi]. For q10/q90 the target is 0.80 --
    materially below that means the interval is lying about its own confidence.
    """
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= np.asarray(lo, dtype=float)) & (y_true <= np.asarray(hi, dtype=float))
    return float(np.mean(inside))
