# model_quantile.py
"""
Quantile regression: predict a distribution instead of a single number.

Two separate reasons this matters, and the first is easy to miss:

1. OBJECTIVE ALIGNMENT. Our headline metric is MAE, which is minimized by the
   conditional MEDIAN. The default `reg:squarederror` objective optimizes the
   conditional MEAN. Training at alpha=0.5 therefore optimizes the thing we
   actually report, so q50 is expected to beat the squared-error model on MAE
   even before anyone looks at intervals.

2. REAL INTERVALS. The GUI currently shows "prediction +/- the model's historical
   MAE", which is a flat band with no coverage guarantee and no sensitivity to an
   individual player. q10/q90 are learned per player from his own features, so a
   boom/bust deep threat gets a wider band than a steady target hog.

XGBoost fits all requested quantiles in ONE model (verified: predict() returns an
(n, len(alphas)) array with q10 <= q50 <= q90 already monotonic), so the extra
quantiles cost nothing beyond the single fit.
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
