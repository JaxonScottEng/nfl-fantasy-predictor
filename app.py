# app.py
"""
Thin entry point. Page config, sidebar controls, and dispatch into views.

Adding a view = add a file in gui/views/ exposing render(position, week, data),
then add one line to VIEWS below. Nothing else changes.
"""
import streamlit as st

from gui import data_access
from gui.views import past_results, player_season, upcoming_week

# id -> render function. The one place a new view gets registered.
VIEWS = {
    "Past Results": past_results.render,
    "Upcoming Week": upcoming_week.render,
    "Player Season": player_season.render,
}

st.set_page_config(page_title="NFL Fantasy Predictor", page_icon="🏈", layout="wide")
st.title("NFL Fantasy Predictor")

with st.sidebar:
    st.header("Controls")

    view_name = st.radio("View", list(VIEWS), index=0)

    # Positions come from config.ACTIVE_POSITIONS -- new positions appear here
    # automatically, nothing to hardcode.
    position = st.selectbox("Position", data_access.get_active_positions())

    weeks = data_access.get_available_weeks(position)
    week = st.selectbox("Week (past results)", weeks,
                        index=len(weeks) - 1 if weeks else 0) if weeks else None

    st.divider()

    if st.button("Refresh Data", width='stretch',
                 help="Re-pull nflverse data and rebuild the feature table."):
        with st.spinner("Pulling data and rebuilding features..."):
            data_access.refresh_data()
        st.success("Data refreshed.")

    if st.button("Re-run Training", width='stretch',
                 help="Walk-forward retrain; regenerates past predictions. Takes minutes."):
        with st.spinner("Running walk-forward training..."):
            data_access.run_training()
        st.success("Training complete.")

    data_ts, preds_ts = data_access.get_last_updated()
    st.caption(f"Data updated: {data_ts:%Y-%m-%d %H:%M}" if data_ts else "Data: not cached yet")
    st.caption(f"Predictions: {preds_ts:%Y-%m-%d %H:%M}" if preds_ts else "Predictions: not generated yet")

data = {
    "predictions": data_access.load_predictions(),
    "last_updated": data_ts,
}

VIEWS[view_name](position, week, data)
