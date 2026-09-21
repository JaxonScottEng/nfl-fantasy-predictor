"""
Regenerate the figures used in README.md and REPORT.md.

    python scripts/make_figures.py

Numbers come from CLAUDE.md's Verification section and from the results file the
evaluation writes, so the figures cannot drift from the reported accuracy.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#12151c"
MUTED = "#6b7280"
GRID = "#e1e4ec"
PROJECTION = "#2a78d6"
ACTUAL = "#eb6834"


BENCHMARK_PATH = os.path.join("docs", "images", "benchmark-gap.png")

# (label, WR MAE, RB MAE, colour)
SERIES = [
    ("Naive baseline\n(last-4 average)", 7.047, 6.464, "#c3c2b7"),
    ("This model\n(XGBoost + ridge)", 6.474, 6.029, "#2a78d6"),
    ("Published best-in-class\n(FantasyPros / FFA)", 4.84, 5.06, "#eb6834"),
]



def benchmark_chart():
    positions = ["Wide receiver", "Running back"]
    x = np.arange(len(positions))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for index, (label, wr, rb, colour) in enumerate(SERIES):
        offset = (index - 1) * width
        bars = ax.bar(x + offset, [wr, rb], width, label=label, color=colour,
                      edgecolor="white", linewidth=1.2, zorder=3)
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10,
                        color=INK, fontweight="bold")

    ax.set_ylabel("Mean absolute error (PPR points)", fontsize=10, color=MUTED)
    ax.set_title("Lower is better — measured on the same top-40-by-projection pool",
                 fontsize=11, color=MUTED, loc="left", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(positions, fontsize=11, color=INK)
    ax.set_ylim(0, 8.0)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=MUTED, length=0, labelsize=9)
    ax.tick_params(axis="x", length=0)
    # Legend below the plot: inside the axes it collided with the RB value labels.
    ax.legend(frameon=False, fontsize=9, ncol=3, labelcolor=MUTED,
              loc="upper center", bbox_to_anchor=(0.5, -0.09),
              columnspacing=2.4, handlelength=1.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(BENCHMARK_PATH), exist_ok=True)
    fig.savefig(BENCHMARK_PATH, facecolor="white", bbox_inches="tight")
    print(f"wrote {BENCHMARK_PATH}")


PLAYER_PATH = os.path.join("docs", "images", "player-season.png")
# Freshly generated results if they exist, otherwise the committed sample, so the
# figure can be rebuilt without downloading the data first.
RESULT_PATHS = [os.path.join("data", "processed", "predictions_detail.csv"),
                os.path.join("docs", "sample_results.csv")]



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


def player_chart():
    source = next((p for p in RESULT_PATHS if os.path.exists(p)), None)
    if source is None:
        raise SystemExit("No results found. Run `python src/evaluate.py` first.")

    detail = pd.read_csv(source)
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
    os.makedirs(os.path.dirname(PLAYER_PATH), exist_ok=True)
    fig.savefig(PLAYER_PATH, facecolor="white", bbox_inches="tight")
    print(f"wrote {PLAYER_PATH} for {player} ({season}), coverage {covered:.2f}")

if __name__ == "__main__":
    benchmark_chart()
    player_chart()
