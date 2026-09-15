# gui/views/past_results.py
"""Past week: predictions vs actual, from train.py's walk-forward output."""
import altair as alt
import streamlit as st

from gui import data_access

SERIES = "#2a78d6"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MUTED = "#898781"

TABLE_COLS = ["player_display_name", "actual", "baseline_pred", "model_pred",
              "hybrid_pred", "baseline_error", "model_error", "hybrid_error"]


def _calibration_chart(rows):
    lo = float(min(rows["actual"].min(), rows["hybrid_pred"].min()))
    hi = float(max(rows["actual"].max(), rows["hybrid_pred"].max()))
    domain = [lo - 1, hi + 1]

    # Neutral reference line: hybrid_pred == actual (perfect prediction).
    reference = (
        alt.Chart(alt.Data(values=[{"x": lo - 1}, {"x": hi + 1}]))
        .mark_line(color=AXIS, strokeDash=[4, 4], strokeWidth=2)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=domain)),
            y=alt.Y("x:Q", scale=alt.Scale(domain=domain)),
        )
    )

    points = (
        alt.Chart(rows)
        .mark_circle(size=90, color=SERIES, opacity=0.85,
                     stroke="#fcfcfb", strokeWidth=2)
        .encode(
            x=alt.X("hybrid_pred:Q", title="Predicted (hybrid)",
                    scale=alt.Scale(domain=domain),
                    axis=alt.Axis(gridColor=GRID, domainColor=AXIS, labelColor=MUTED,
                                  titleColor=MUTED, tickColor=AXIS)),
            y=alt.Y("actual:Q", title="Actual PPR points",
                    scale=alt.Scale(domain=domain),
                    axis=alt.Axis(gridColor=GRID, domainColor=AXIS, labelColor=MUTED,
                                  titleColor=MUTED, tickColor=AXIS)),
            tooltip=[
                alt.Tooltip("player_display_name:N", title="Player"),
                alt.Tooltip("actual:Q", title="Actual", format=".1f"),
                alt.Tooltip("hybrid_pred:Q", title="Hybrid", format=".1f"),
                alt.Tooltip("baseline_pred:Q", title="Baseline", format=".1f"),
            ],
        )
    )

    return (reference + points).properties(height=380).configure_view(strokeWidth=0)


def render(position, week, data):
    st.subheader(f"{position} — week {week}: predictions vs actual")

    rows = data_access.get_week_predictions(position, week)
    if len(rows) == 0:
        st.info(
            "No stored predictions for this position/week. Use **Re-run Training** "
            "in the sidebar to regenerate `data/processed/predictions_detail.csv`."
        )
        return

    st.caption(
        f"{len(rows)} players. Out-of-sample walk-forward predictions: each week was "
        "predicted by a model trained only on earlier games."
    )

    # Baseline shown next to every model result (CLAUDE.md hard rule).
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline MAE", f"{rows['baseline_error'].mean():.3f}")
    c2.metric("Model MAE", f"{rows['model_error'].mean():.3f}")
    c3.metric("Hybrid MAE", f"{rows['hybrid_error'].mean():.3f}")

    st.altair_chart(_calibration_chart(rows), width='stretch')
    st.caption(
        "Points above the dashed line were under-predicted; below it, over-predicted."
    )

    st.dataframe(
        rows[TABLE_COLS].sort_values("actual", ascending=False),
        width='stretch',
        hide_index=True,
    )
