# NFL Fantasy Points Predictor

Weekly PPR projections for NFL wide receivers and running backs — built end to end with
Claude Code, and measured against commercial projections on their own terms. The headline
result is that **it loses**: 6.47 MAE for WR against the published field's 4.84–4.94. The
interesting part is how that number was arrived at, and what it cost to find out.

> **Honest summary:** 8% (WR) and 7% (RB) better than a naive baseline; 34% (WR) and 19%
> (RB) worse than commercial projections. Measured on 21,600 walk-forward-scored
> player-weeks using the same player-pool convention published accuracy studies use.

---

## How it was built: AI-assisted engineering

The project had two goals — advance my machine learning practice, and learn to direct an AI
agent on work where being wrong is easy and hard to notice. The second turned out to be
mostly about **building machinery that catches the agent, and me, being wrong.**

Four artifacts, each of which is a decision I made rather than code an agent produced:

**1. A regression gate that blocks the agent.**
[`.claude/hooks/check_wr_regression.py`](.claude/hooks/check_wr_regression.py) runs after
*any* edit to `src/*.py`. It re-runs the evaluation, parses the per-position MAE, compares
against six locked-in constants, and exits non-zero on drift — which stops the agent and
forces an explanation. Its instruction is deliberately asymmetric:

> *"Investigate before proceeding — treat unexplained changes (even improvements) as a
> leakage suspect."*

An agent that can silently improve your metrics is more dangerous than one that breaks
them. It fired for real during development ([LOG.md](LOG.md), Phase 9) when a change moved
the numbers, and it has blocked every unexplained movement since.

**2. A negative result I made the agent keep.**
Injury features were an obvious win, so I had them built. They measured null
([LOG.md](LOG.md), Phase 11) — because a player listed "Out" has **no stats row at all**
(verified 996 of 996), so the error they targeted cannot occur in this dataset. They were
reverted rather than quietly retained. The best consequence: the project's strictest
leakage rule did not have to be relaxed to accommodate a feature that bought nothing.

**3. I shipped the second-best model on purpose.**
A three-way ensemble won on MAE (6.427 vs 6.474). I rejected it: checked on a real player,
it projected an elite WR at 18.4 where the model said 22.7 — a −4.4 point bias hidden
inside a −0.84 average, because a median sits below the mean on right-skewed scoring. A
0.7% metric gain was not worth a projection system that under-sells its best players.

**4. The harness that invalidated my own headline.**
I reported 4.49 MAE for months. Building the measurement layer revealed it was a pool
artifact — scoring ~140 WRs a week instead of the top 40 flatters MAE by ~30%. The same
model, measured honestly, scores 6.47. The number got worse and the project got better.

The tests in this repo were written last, and immediately caught a regression that had been
sitting in `check_fidelity()` undetected — the exact failure mode the hook was built for,
in the one code path the hook did not cover.

## Results

![Accuracy against published projections](docs/images/benchmark-gap.png)

| | Naive baseline | **This model** | Published best-in-class |
|---|---|---|---|
| **WR** | 7.047 | **6.474** | 4.84–4.94 |
| **RB** | 6.464 | **6.029** | 5.06–5.20 |

Mean absolute error in PPR points, lower is better. Top-40-by-projection pool, 2021–2025,
180 walk-forward folds, 21,600 scored rows. Published figures: Fantasy Football Analytics'
11-season study and FantasyPros, measured on the same pool convention.

**Why the pool matters more than the model.** MAE is meaningless without a declared player
pool. Scoring every rostered WR includes ~100 bench players a week who score near zero and
are trivially predictable, which drags the average down by roughly 30%. Any projection
accuracy figure quoted without its pool — including the one this project published for
months — is not comparable to anything.

Prediction intervals are calibrated rather than asserted: the model's own 10th–90th
percentile band achieves **0.78 (WR) / 0.79 (RB)** coverage against an 0.80 target.

![A player's season against the projected range](docs/images/player-season.png)

## What it does

A Streamlit app with three views:

- **Past Results** — walk-forward predictions against actuals for any position and week,
  with the naive baseline shown alongside every model number.
- **Upcoming Week** — projections for the next unplayed week, with prediction intervals.
- **Player Season** — one player's whole season: actual points where played, projection and
  range for every week, including weeks not yet played.

## Architecture

```
config.py            single source of truth: seasons, positions, windows, splits
data_load.py         nflverse pull, parquet cache
features_*.py        usage, opportunity, efficiency, snaps, expected points, opponent, market
build_features.py    joins everything, asserts no row inflation
validation.py        walk-forward fold generator
registry.py          id -> function registry
  metrics.py         MAE, RMSE, R2, bias, Spearman, hit rate, consistency
  pools.py           player-pool definitions -- the comparability story
  comparators.py     anything scoreable: baseline, models, ensembles
evaluate.py          the harness: row-level predictions, metrics, fidelity check
train.py / upcoming.py / player_timeline.py
app.py + gui/        Streamlit, views registered in one dict
```

Metrics, pools, comparators and GUI range methods all extend the same way: write a
decorated function, change no callers.

## Validation protocol

- **Walk-forward only.** Train on games strictly before week W, predict W, refit per fold.
  No random splits — a shuffled split on time-series data leaks the future.
- **Every feature lagged at source**: `.shift(1)` then `.rolling()`, grouped by player.
  Ratios are built from already-lagged columns.
- **Windows follow a player's game sequence**, not week number, so byes don't shift them.
- **Regular season only**, filtered *before* rolling — filtering only prediction targets
  still leaves playoff games inside the windows.
- **The baseline is scored on identical rows** as every model, in the same fold and pool.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate elsewhere
pip install -r requirements.txt

python src/data_load.py      # download + cache nflverse data (~25 MB, first run only)
python src/evaluate.py       # fidelity check + accuracy table
streamlit run app.py         # the GUI
```

Run every command from the project root — cache paths are relative. A full multi-season
evaluation takes several minutes because it refits a model per walk-forward fold by design.

## Tests

```bash
pytest              # fast tier: 62 invariant tests, ~11s, no data needed
pytest -m slow      # fidelity gate against real data (minutes)
```

The fast tier is structural and invariant-based: leakage properties, feature-lag
correctness, pool definitions, the join guard, metric sign conventions, and an end-to-end
harness run on synthetic data. Numeric model outputs are deliberately *not* asserted there
— the regression hook already owns that job, and duplicating it would produce a brittle
suite.

## Limitations

- WR and RB only; QB/TE are straightforward extensions, K/DST need a new target entirely.
- The 2024 test season is reserved and has never been used as a final holdout.
- Multi-week-ahead projections are a current-form outlook, not a forecast — lagged features
  cannot advance until the intervening games are played.
- The remaining gap to commercial projections is mostly a **data** gap (player prop markets,
  real availability signal), not a modelling one.

Full detail, including every measured negative result, is in **[REPORT.md](REPORT.md)**.
The complete phase-by-phase build history is in **[LOG.md](LOG.md)**.

## License

MIT — see [LICENSE](LICENSE).
