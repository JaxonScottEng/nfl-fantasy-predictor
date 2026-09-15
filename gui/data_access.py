# gui/data_access.py
"""
The ONLY module in the GUI that touches the pipeline.

Views import from here, never from train.py / build_features.py / upcoming.py
directly -- so if the pipeline changes, this file is the single thing to update.

Nothing here reimplements pipeline logic; every function wraps an existing
function from src/.
"""
import os
import sys
import datetime

import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# The pipeline uses paths relative to the project root (data_load.CACHE_PATH is
# "data/raw/..."). Streamlit's cwd is wherever `streamlit run` was invoked, so
# without this the pipeline silently reads/writes a DIFFERENT cache directory.
os.chdir(PROJECT_ROOT)

import config                                     # noqa: E402
import train                                      # noqa: E402
import upcoming as upcoming_pipeline              # noqa: E402
from data_load import load_weekly                 # noqa: E402
from build_features import build_full_feature_table  # noqa: E402

RAW_CACHE = os.path.join(PROJECT_ROOT, "data", "raw", "weekly_stats.parquet")
PREDICTIONS_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "predictions_detail.csv")


def get_active_positions():
    """Positions come from config -- never hardcoded, so new ones just appear."""
    return list(config.ACTIVE_POSITIONS)


def get_hybrid_threshold(position):
    return train.HYBRID_THRESHOLD_BY_POSITION.get(
        position, train.DEFAULT_HYBRID_THRESHOLD
    )


@st.cache_data(show_spinner=False)
def load_predictions():
    """
    train.py's walk-forward output (data/processed/predictions_detail.csv).
    Reused as-is -- the GUI never re-derives past predictions.
    """
    if not os.path.exists(PREDICTIONS_CSV):
        return pd.DataFrame()
    return pd.read_csv(PREDICTIONS_CSV)


def get_available_weeks(position):
    preds = load_predictions()
    if len(preds) == 0 or "position" not in preds.columns:
        return []
    weeks = preds.loc[preds["position"] == position, "week"].unique()
    return sorted(int(w) for w in weeks)


def get_week_predictions(position, week):
    preds = load_predictions()
    if len(preds) == 0:
        return preds
    return preds[(preds["position"] == position) & (preds["week"] == week)].copy()


@st.cache_data(show_spinner=False)
def get_segment_mae(position):
    """
    The hybrid model's REAL historical MAE for this position, split by volume
    segment. Feeds the approximate prediction range. None if no predictions yet.
    """
    preds = load_predictions()
    if len(preds) == 0:
        return None

    pos = preds[preds["position"] == position]
    if len(pos) == 0:
        return None

    threshold = get_hybrid_threshold(position)
    high = pos[pos["baseline_pred"] >= threshold]
    low = pos[pos["baseline_pred"] < threshold]

    return {
        "threshold": threshold,
        "high": float(high["hybrid_error"].mean()) if len(high) else None,
        "low": float(low["hybrid_error"].mean()) if len(low) else None,
        "n_high": len(high),
        "n_low": len(low),
    }


@st.cache_data(show_spinner=False)
def load_upcoming(position):
    """Next unplayed week's predictions. Returns (DataFrame, meta dict)."""
    return upcoming_pipeline.predict_upcoming(position)


@st.cache_data(show_spinner=False)
def load_feature_table():
    return build_full_feature_table()


def refresh_data():
    """
    Explicit, user-triggered: re-pull from nflverse and rebuild features.
    Never runs on page load.
    """
    load_weekly(force_refresh=True)
    build_full_feature_table()
    st.cache_data.clear()


def run_training():
    """
    Re-run walk-forward training, regenerating predictions_detail.csv.
    Slow (minutes) -- kept separate from refresh_data for that reason.
    """
    train.run_walk_forward_training(save_details=True)
    st.cache_data.clear()


def _mtime(path):
    if not os.path.exists(path):
        return None
    return datetime.datetime.fromtimestamp(os.path.getmtime(path))


def get_last_updated():
    """(raw data timestamp, predictions timestamp) -- either may be None."""
    return _mtime(RAW_CACHE), _mtime(PREDICTIONS_CSV)
