# gui/prediction_ranges.py
"""
Prediction ranges, as a swappable method registry.

Adding a new method (e.g. Monte Carlo) means:
  1. write a _your_method(player_row, **kwargs) -> (low, high) function here
  2. add it to _METHODS, METHOD_LABELS and METHOD_CAVEATS
No view code changes -- views discover methods via available_methods().
"""

MAE_APPROX = "mae_approx"
QUANTILE = "quantile"


def _mae_approx_range(player_row, segment_mae=None, point_col="model_pred", **_):
    """
    Point prediction +/- the model's historical MAE for this player's volume
    segment (high/low, split at the position's hybrid threshold).

    This is a rough spread, NOT a calibrated interval: MAE is an average error
    across many players, so it neither adapts to an individual player's own
    variance nor carries a coverage guarantee.
    """
    pred = float(player_row[point_col])
    if not segment_mae:
        return None, None

    baseline = float(player_row.get("baseline_pred", pred))
    band = segment_mae["high"] if baseline >= segment_mae["threshold"] else segment_mae["low"]
    if band is None:
        return None, None

    # Floor at 0: PPR can dip slightly negative on lost fumbles, but a negative
    # projection floor is noise for lineup decisions.
    return max(0.0, pred - band), pred + band


def _quantile_range(player_row, **_):
    """
    The model's own 10th-90th percentile prediction for this player.

    Unlike the MAE band, this is learned per player rather than per segment, so a
    boom/bust deep threat gets a wider interval than a steady target hog with the
    same projection. Returns (None, None) when the quantile columns are absent, so
    a caller working with older prediction data degrades instead of breaking.
    """
    lo, hi = player_row.get("q10"), player_row.get("q90")
    if lo is None or hi is None:
        return None, None
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return None, None
    if lo != lo or hi != hi:      # NaN check without importing pandas
        return None, None
    return max(0.0, lo), hi


# Quantile first so it is the default where available -- it is a real interval,
# whereas the MAE band is a flat approximation kept for comparison.
_METHODS = {
    QUANTILE: _quantile_range,
    MAE_APPROX: _mae_approx_range,
}

METHOD_LABELS = {
    QUANTILE: "Quantile (10th-90th percentile)",
    MAE_APPROX: "Approximate (+/- historical MAE)",
}

METHOD_CAVEATS = {
    QUANTILE: (
        "Learned interval: the model's own 10th-90th percentile prediction for this "
        "player, so it widens for volatile players rather than applying one flat band. "
        "Targets 80% coverage; see LOG.md for the measured figure. Still a model "
        "output, not a guarantee -- weekly fantasy scoring is mostly irreducible noise."
    ),
    MAE_APPROX: (
        "Approximation, not a confidence interval. This is the point prediction "
        "plus/minus the model's average absolute error for this position and volume "
        "segment, measured on walk-forward validation. It has no coverage guarantee "
        "and does not adapt to an individual player's week-to-week variance."
    ),
}


def available_methods():
    return list(_METHODS)


def get_prediction_range(player_row, method=MAE_APPROX, **kwargs):
    """Returns (low, high), or (None, None) if the method lacks inputs."""
    if method not in _METHODS:
        raise ValueError(
            f"Unknown range method {method!r}. Available: {available_methods()}"
        )
    return _METHODS[method](player_row, **kwargs)


def add_range_columns(df, method=MAE_APPROX, **kwargs):
    """Convenience: adds range_low/range_high so views stay loop-free."""
    if len(df) == 0:
        return df

    out = df.copy()
    bounds = [get_prediction_range(row, method=method, **kwargs)
              for _, row in out.iterrows()]
    out["range_low"] = [b[0] for b in bounds]
    out["range_high"] = [b[1] for b in bounds]
    return out
