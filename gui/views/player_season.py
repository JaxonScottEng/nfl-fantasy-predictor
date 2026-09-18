# gui/views/player_season.py
"""One player's season: actual points per week against the projected range."""
import altair as alt
import pandas as pd
import streamlit as st

from gui import data_access

PROJECTION = "#2a78d6"
ACTUAL = "#eb6834"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MUTED = "#898781"

TABLE_COLS = ["week", "opponent_team", "actual", "ensemble_pred",
              "q10", "q90", "baseline_pred"]


def _season_chart(rows, player_name, first_unplayed):
    weeks = rows["week"].tolist()
    domain = [min(weeks) - 0.5, max(weeks) + 0.5]
    x = alt.X("week:Q", title="Week", scale=alt.Scale(domain=domain, nice=False),
              axis=alt.Axis(tickMinStep=1, gridColor=GRID, domainColor=AXIS,
                            labelColor=MUTED, titleColor=MUTED, tickColor=AXIS))
    y_axis = alt.Axis(gridColor=GRID, domainColor=AXIS, labelColor=MUTED,
                      titleColor=MUTED, tickColor=AXIS)

    has_band = rows["q10"].notna().any()
    layers = []

    if has_band:
        layers.append(
            alt.Chart(rows).mark_area(color=PROJECTION, opacity=0.16).encode(
                x=x,
                y=alt.Y("q10:Q", title="PPR points", axis=y_axis),
                y2="q90:Q",
            )
        )

    layers.append(
        alt.Chart(rows).mark_line(color=PROJECTION, strokeWidth=2).encode(
            x=x, y=alt.Y("ensemble_pred:Q", title="PPR points", axis=y_axis))
    )
    layers.append(
        alt.Chart(rows).mark_point(color=PROJECTION, size=80, filled=True,
                                   stroke="#fcfcfb", strokeWidth=1.5).encode(
            x=x,
            y=alt.Y("ensemble_pred:Q", title="PPR points", axis=y_axis),
            tooltip=[
                alt.Tooltip("week:Q", title="Week"),
                alt.Tooltip("opponent_team:N", title="Opponent"),
                alt.Tooltip("ensemble_pred:Q", title="Projected", format=".1f"),
                alt.Tooltip("q10:Q", title="Range low", format=".1f"),
                alt.Tooltip("q90:Q", title="Range high", format=".1f"),
                alt.Tooltip("actual:Q", title="Actual", format=".1f"),
            ],
        )
    )

    played = rows[rows["actual"].notna()]
    if len(played):
        layers.append(
            alt.Chart(played).mark_point(color=ACTUAL, size=130, filled=True,
                                         stroke="#fcfcfb", strokeWidth=1.5).encode(
                x=x,
                y=alt.Y("actual:Q", title="PPR points", axis=y_axis),
                tooltip=[
                    alt.Tooltip("week:Q", title="Week"),
                    alt.Tooltip("opponent_team:N", title="Opponent"),
                    alt.Tooltip("actual:Q", title="Actual", format=".1f"),
                    alt.Tooltip("ensemble_pred:Q", title="Projected", format=".1f"),
                ],
            )
        )

    # Mark where the season stops being history and starts being outlook.
    if first_unplayed is not None and first_unplayed > min(weeks):
        boundary = pd.DataFrame({"week": [first_unplayed - 0.5]})
        layers.append(
            alt.Chart(boundary).mark_rule(color=AXIS, strokeDash=[4, 4],
                                          strokeWidth=2).encode(x="week:Q")
        )

    return (alt.layer(*layers)
            .properties(height=360, title=f"{player_name} — actual vs projected")
            .configure_view(strokeWidth=0)
            .configure_title(color=MUTED, fontSize=13, anchor="start"))


def render(position, week, data):
    """`week` is ignored -- this view always shows a whole season."""
    seasons = data_access.get_timeline_seasons()
    if not seasons:
        st.info("No seasons available.")
        return

    col_a, col_b = st.columns([1, 2])
    season = col_a.selectbox("Season", seasons, index=len(seasons) - 1)

    players = data_access.get_players(position, season)
    if not players:
        st.info(f"No {position} players with data in {season}.")
        return
    player = col_b.selectbox("Player", players)

    with st.spinner(f"Building {player}'s {season} timeline..."):
        rows, meta = data_access.load_player_timeline(position, season, player)

    if len(rows) == 0:
        st.info(f"No weeks to show for {player} in {season}.")
        return

    st.altair_chart(_season_chart(rows, player, meta.get("first_unplayed_week")),
                    width='stretch')

    played = rows[rows["actual"].notna()]
    upcoming = rows[rows["actual"].isna()]
    c1, c2, c3 = st.columns(3)
    c1.metric("Weeks played", len(played))
    if len(played):
        c2.metric("Actual per week", f"{played['actual'].mean():.1f}")
        c3.metric("Projection MAE", f"{(played['actual'] - played['ensemble_pred']).abs().mean():.2f}")

    st.caption(
        "Blue line and band: projection with its 10th–90th percentile range. "
        "Orange: what he actually scored. Dashed line marks where played weeks end."
        + (f" Weeks {int(upcoming['week'].min())}+ are projected but not yet played."
           if len(upcoming) else "")
    )

    if len(upcoming) and meta.get("outlook_weeks"):
        st.warning(
            f"Weeks beyond {meta['first_unplayed_week']} are a current-form outlook, "
            "not a game-specific projection: rolling features cannot advance until "
            "those games are played, and most future games have no betting line "
            "posted yet. They vary only by opponent.",
            icon="⚠️",
        )

    st.dataframe(rows[[c for c in TABLE_COLS if c in rows.columns]],
                 width='stretch', hide_index=True)
