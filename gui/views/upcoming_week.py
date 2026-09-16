# gui/views/upcoming_week.py
"""Next unplayed week's predictions, with an approximate range."""
import altair as alt
import streamlit as st

from gui import data_access, prediction_ranges

SERIES = "#2a78d6"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MUTED = "#898781"

TOP_N = 15
POINT_COL = "model_pred"
TABLE_COLS = ["player_display_name", "team", "opponent_team", "baseline_pred",
              "model_pred", "range_low", "range_high", "implied_total"]


def _projection_chart(rows):
    ranked = rows.head(TOP_N)
    order = ranked["player_display_name"].tolist()

    base = alt.Chart(ranked).encode(
        y=alt.Y("player_display_name:N", sort=order, title=None,
                axis=alt.Axis(labelColor=MUTED, domainColor=AXIS, tickColor=AXIS)),
    )

    has_range = ranked["range_low"].notna().any()
    layers = []

    if has_range:
        layers.append(
            base.mark_rule(color=AXIS, strokeWidth=2).encode(
                x=alt.X("range_low:Q", title="Projected PPR points",
                        axis=alt.Axis(gridColor=GRID, domainColor=AXIS,
                                      labelColor=MUTED, titleColor=MUTED, tickColor=AXIS)),
                x2="range_high:Q",
            )
        )

    layers.append(
        base.mark_circle(size=110, color=SERIES, opacity=0.9,
                         stroke="#fcfcfb", strokeWidth=2).encode(
            x=alt.X(f"{POINT_COL}:Q", title="Projected PPR points",
                    axis=alt.Axis(gridColor=GRID, domainColor=AXIS,
                                  labelColor=MUTED, titleColor=MUTED, tickColor=AXIS)),
            tooltip=[
                alt.Tooltip("player_display_name:N", title="Player"),
                alt.Tooltip("opponent_team:N", title="Opponent"),
                alt.Tooltip(f"{POINT_COL}:Q", title="Projection", format=".1f"),
                alt.Tooltip("baseline_pred:Q", title="Baseline", format=".1f"),
                alt.Tooltip("range_low:Q", title="Range low", format=".1f"),
                alt.Tooltip("range_high:Q", title="Range high", format=".1f"),
            ],
        )
    )

    return (
        alt.layer(*layers)
        .properties(height=28 * len(ranked) + 40)
        .configure_view(strokeWidth=0)
    )


def render(position, week, data):
    """`week` is ignored: this view always targets the next UNPLAYED week."""
    with st.spinner(f"Building {position} projections for the upcoming week..."):
        rows, meta = data_access.load_upcoming(position)

    if len(rows) == 0:
        reason = meta.get("reason", "unknown")
        if reason == "no unplayed week":
            st.info(
                "Every season in `config.SEASONS` is complete, so there is no "
                "upcoming week to predict. Extend `SEASONS` in `src/config.py` to "
                "include the current season, then use **Refresh Data**."
            )
        else:
            st.info(f"No {position} projections available for the upcoming week ({reason}).")
        return

    st.subheader(f"{position} — {meta['season']} week {meta['week']} projections")
    st.caption(
        f"{meta['n_players']} players, model trained on {meta['n_train_rows']:,} "
        "completed player-games strictly before this week. Features use only prior "
        "games, so no same-week data leaks in."
    )

    methods = prediction_ranges.available_methods()
    method = st.selectbox(
        "Range method",
        methods,
        format_func=lambda m: prediction_ranges.METHOD_LABELS.get(m, m),
    )

    segment_mae = data_access.get_segment_mae(position)
    if segment_mae is None:
        st.warning(
            "No stored validation errors yet, so ranges can't be computed. "
            "Use **Re-run Training** in the sidebar."
        )
        rows["range_low"] = None
        rows["range_high"] = None
    else:
        rows = prediction_ranges.add_range_columns(
            rows, method=method, segment_mae=segment_mae
        )
        st.warning(prediction_ranges.METHOD_CAVEATS.get(method, ""), icon="⚠️")

    st.altair_chart(_projection_chart(rows), width='stretch')
    st.caption(f"Top {min(TOP_N, len(rows))} by model projection. Bars show the approximate range.")

    st.dataframe(rows[TABLE_COLS], width='stretch', hide_index=True)
