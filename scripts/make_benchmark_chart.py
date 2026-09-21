"""
Generate docs/images/benchmark-gap.png -- this project against published accuracy.

Deliberately plots the unflattering comparison. The naive baseline is included
because "beats a naive baseline" is the weak claim most projection projects stop
at, and showing both makes clear which bar actually matters.

Numbers are the locked-in figures from CLAUDE.md's Verification section: top-40
by projection, 2021-2025, 180 walk-forward folds, 21,600 scored rows. Published
figures are Fantasy Football Analytics' 11-season study and FantasyPros' recent
seasons, measured on the same pool convention.

    python scripts/make_benchmark_chart.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = os.path.join("docs", "images", "benchmark-gap.png")

# (label, WR MAE, RB MAE, colour)
SERIES = [
    ("Naive baseline\n(last-4 average)", 7.047, 6.464, "#c3c2b7"),
    ("This model\n(XGBoost + ridge)", 6.474, 6.029, "#2a78d6"),
    ("Published best-in-class\n(FantasyPros / FFA)", 4.84, 5.06, "#eb6834"),
]

INK = "#12151c"
MUTED = "#6b7280"
GRID = "#e1e4ec"


def main():
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
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, facecolor="white", bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
