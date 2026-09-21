"""
Generate docs/images/player-season.png -- one player's season, projection vs actual.

Uses the real walk-forward predictions in data/processed/predictions_detail.csv:
every point was produced by a model trained only on earlier games. The shaded band
is the model's own 10th-90th percentile prediction, which is what makes the
calibration claim (measured 0.78-0.79 coverage against an 0.80 target) legible.

    python scripts/make_player_chart.py [player name]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DETAIL_PATH = os.path.join("data", "processed", "predictions_detail.csv")
OUT_PATH = os.path.join("docs", "images", "player-season.png")

PROJECTION = "#2a78d6"
ACTUAL = "#eb6834"
INK = "#12151c"
MUTED = "#6b7280"
GRID = "#e1e4ec"


def pick_player(detail, requested=None):
    """
    Choose on PROJECTED volume, never on actual outcome.

    Picking the highest-scoring WR would systematically select the biggest
    over-performer -- precisely the player a well-calibrated interval is most
    likely to miss high -- and make calibration look far worse than it measures.
    Selecting on a pre-outcome variable avoids putting a thumb on the scale in
    either direction; a typical starter is what a reader should see.
    """
    if requested:
        return requested

    wr = detail[detail["position"] == "WR"]
    weeks_played = wr.groupby("player_display_name")["week"].nunique()
    eligible = weeks_played[weeks_played >= 15].index
    projected = (wr[wr["player_display_name"].isin(eligible)]
                 .groupby("player_display_name")["ensemble_pred"].mean())
    # The starter closest to the median projection -- representative by design.
    return (projected - projected.median()).abs().idxmin()


def main():
    if not os.path.exists(DETAIL_PATH):
        raise SystemExit(f"{DETAIL_PATH} missing -- run `python src/evaluate.py` first")

    detail = pd.read_csv(DETAIL_PATH)
    player = pick_player(detail, " ".join(sys.argv[1:]) or None)
    rows = detail[detail["player_display_name"] == player].sort_values("week")
    if len(rows) == 0:
        raise SystemExit(f"no rows for {player!r}")

    season = int(rows["season"].iloc[0])

    # Report the position-wide figure, not this player's 18 weeks -- one season of
    # one player is far too small a sample to characterize interval coverage.
    position = rows["position"].iloc[0]
    same_position = detail[detail["position"] == position]
    covered = ((same_position["actual"] >= same_position["q10"])
               & (same_position["actual"] <= same_position["q90"])).mean()

    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.fill_between(rows["week"], rows["q10"], rows["q90"], color=PROJECTION,
                    alpha=0.15, zorder=2, label="Projected range (10th–90th pct)")
    ax.plot(rows["week"], rows["ensemble_pred"], color=PROJECTION, linewidth=2,
            zorder=3, label="Projection")
    ax.scatter(rows["week"], rows["ensemble_pred"], color=PROJECTION, s=26,
               zorder=4, edgecolor="white", linewidth=1)
    ax.scatter(rows["week"], rows["actual"], color=ACTUAL, s=64, zorder=5,
               edgecolor="white", linewidth=1.2, label="Actual points")

    ax.set_xlabel("Week", fontsize=10, color=MUTED)
    ax.set_ylabel("PPR points", fontsize=10, color=MUTED)
    ax.set_title(f"{player} — {season} walk-forward projections vs actual\n"
                 f"Across all {position} weeks, {covered:.0%} of actuals landed "
                 f"inside the projected range (target 80%)",
                 fontsize=11, color=INK, loc="left", pad=12)
    ax.set_xticks(sorted(rows["week"].unique()))
    ax.set_ylim(bottom=0)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.legend(frameon=False, fontsize=9, ncol=3, labelcolor=MUTED,
              loc="upper center", bbox_to_anchor=(0.5, -0.14))

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, facecolor="white", bbox_inches="tight")
    print(f"wrote {OUT_PATH} for {player} ({season}), coverage {covered:.2f}")


if __name__ == "__main__":
    main()
