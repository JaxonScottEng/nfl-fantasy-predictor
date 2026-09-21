# NFL Fantasy Predictor

## Overview

Weekly PPR fantasy-points predictor using nflverse data. WR and RB currently,
one separate model per position — never pooled. Adding a position means: add it
to `config.ACTIVE_POSITIONS`, give it a feature list in `train.py`'s
`FEATURE_COLS_BY_POSITION`, and check that any position-aggregated feature
(e.g. opponent defense) is grouped per position rather than pooled.

## Stack

- Language: Python 3.13, virtualenv (`.venv`)
- Data: `nflreadpy` (returns Polars — convert to pandas at load boundary)
- Modeling: pandas, scikit-learn, XGBoost
- Package manager: pip (`requirements.txt`)

## Project Structure

- `src/config.py` — single source of truth: seasons, active positions,
  rolling windows, train/validate/test split
- `src/data_load.py` — nflreadpy pull + local parquet cache
- `src/build_target.py` — filters to active position(s), sorts by player/game
- `src/baselines.py` — lagged rolling-average baseline
- `src/validation.py` — walk-forward fold generator + leakage assertion
- `src/features_player.py` — lagged usage/efficiency rolling features
- `src/features_opponent.py` — lagged opponent-defense rolling features
- `src/features_context.py` — schedule/Vegas join, team-code normalization
- `src/features_snap.py` — offensive snap share (bridges pfr_id→player_id via rosters)
- `src/features_expected.py` — nflverse expected fantasy points; the single strongest
  signal in the model (40% of WR importance). Lagged only — a same-week expected
  value would be an oracle, not a projection
- `src/build_features.py` — joins player + defense + context into one table
- `src/train.py` — walk-forward training: baseline vs model vs hybrid
- `src/registry.py` — generic id→function registry; the extension seam used by
  metrics, pools and comparators (and mirrored by `gui/prediction_ranges.py`)
- `src/metrics.py` — accuracy metrics registry (MAE, RMSE, R², bias, Spearman,
  top-N hit rate, CoV of weekly MAE). `train.py` imports `mae`/`rmse` from here
- `src/pools.py` — player-pool definitions; this is what makes results comparable
- `src/comparators.py` — anything scoreable, one per function. Defines
  `SHIPPED_COMPARATOR_ID`, the single source of truth for what users see
- `src/model_quantile.py` — quantile regression: q50 optimizes MAE directly (the
  metric we report), q10/q90 give a real per-player interval
- `src/evaluate.py` — the harness: `run_evaluation()` → row-level predictions,
  `summarize()` → metrics, `check_fidelity()` → proves it matches `train.py`,
  `write_prediction_details()` → the CSV the GUI reads
- `src/player_timeline.py` — a season week-by-week: walk-forward for played weeks,
  form outlook for unplayed ones, with the boundary reported in the metadata
- `gui/views/player_season.py` — one player's season, actual vs projected range
- `src/upcoming.py` — predictions for a not-yet-played week (placeholder-row
  trick: reuses the existing lagged feature builders, so no leakage)
- `src/error_analysis.py` — segment-level error breakdown
- `app.py` — Streamlit entry point: sidebar + `VIEWS` registry, stays thin
- `gui/data_access.py` — the only GUI module that touches the pipeline
- `gui/prediction_ranges.py` — range methods as a swappable registry
- `gui/views/` — one file per view, each exposing `render(position, week, data)`

## Commands

- `python src/data_load.py` — pull/cache nflverse data
- `python src/build_features.py` — build full feature table, check null counts
- `python src/train.py` — run walk-forward training, print MAE/RMSE comparison
- `python src/validation.py` — run fold generator standalone, verify no leakage
- `python src/upcoming.py` — print next unplayed week's predictions per position
- `python src/evaluate.py [seasons...]` — fidelity check + top-40 accuracy table
- `python src/evaluate.py --regression` — fast machine-readable numbers (hook uses this)
- `streamlit run app.py` — launch the local GUI
- `pytest` — fast invariant suite (~11s, no data needed). Run this after any change
  to features, validation, pools or the registries
- `pytest -m slow` — fidelity gate against real data (minutes)
- `python scripts/make_benchmark_chart.py` / `make_player_chart.py` — regenerate the
  README images

Run every command from the PROJECT ROOT. `data_load.CACHE_PATH` is relative, so a
different cwd silently reads/writes a different cache.

## Verification

Always run after changes: `python src/evaluate.py`

### The official metric: top-40 pool

Accuracy is measured on the **top 40 WR/RB by projected points per week** (top 20
for QB/TE if added) — the pool convention published accuracy studies use. The old
all-rows numbers are retired: scoring every WR with a prior game (~140/week) includes
deep-bench players who score near zero and are trivial to predict, which made MAE look
~30% better than it is. A number from one pool cannot be compared to a number from
another. See `src/pools.py`.

**Locked-in, 2023 walk-forward, top-40 pool** (what the hook checks):

| | WR MAE | RB MAE |
|---|---|---|
| Baseline (last-4 avg) | 7.258 | 6.129 |
| Model (XGBoost) | 6.806 | 5.746 |
| **Ensemble (shipped)** | **6.759** | **5.758** |

**Multi-season, 2021–2025, top-40 pool** (180 folds, 21,600 scored rows — the
headline figure for any external claim):

| | WR MAE | RB MAE |
|---|---|---|
| Baseline | 7.047 | 6.464 |
| Model (XGBoost) | 6.514 | 6.073 |
| **Ensemble (shipped)** | **6.474** | **6.029** |

Note the single-season and multi-season numbers disagree in direction for WR
(2023 got slightly worse while the 5-season average improved). Judge changes on the
multi-season run; one season of 18 folds is too noisy to conclude from.

Run with `python src/evaluate.py 2021 2022 2023 2024 2025`.

### Where we actually stand

Published best-in-class on the same pool convention: **WR 4.84–4.94, RB 5.06–5.20**
(Fantasy Football Analytics, 11 seasons / 9 sources; FantasyPros recent). We are
roughly **34% behind on WR and 19% behind on RB**. Do not describe this project as
competitive with commercial projections until those gaps close.

### The shipped projection

`comparators.SHIPPED_COMPARATOR_ID` = **`ensemble_xgb_ridge`** — the equal-weight mean
of XGBoost and ridge. Everything user-facing reads that one id. Its q10/q90 interval
comes from a separate quantile model and is measured, not assumed: **coverage 0.78 WR
/ 0.79 RB against an 0.80 target**.

A 3-way ensemble adding the quantile median measured BETTER on MAE (WR 6.427, RB
5.973) and was deliberately rejected. A median sits below the mean on right-skewed
scoring and skew grows with volume, so its bias lands on exactly the players that
matter: about **−4.4 points on an elite WR** versus −0.84 on average. Re-adding it is
one `register_ensemble()` call if that tradeoff is ever wanted.

The hybrid is obsolete and no longer shipped. It existed because the model used to
LOSE to the baseline on low-volume players; after the expected-points features landed
the model wins in both segments, so its baseline fallback was pure drag. Still
registered as a comparator so older comparisons stay reproducible.

Context for what is achievable: best-in-class projections explain only 3–23% of weekly
variance, and WR MAE ~4.8–4.9 is near the practical floor. Chasing a much lower number
is not realistic; the winnable axes are availability handling, rank ordering and
calibrated distributions.

If any locked-in number moves without an intentional cause, stop and investigate before
proceeding — treat unexplained improvement as a leakage suspect, not a win.

These numbers are enforced automatically: a `PostToolUse` hook
(`.claude/hooks/check_wr_regression.py`, registered in `.claude/settings.json`) runs
`python src/evaluate.py --regression` after any edit to `src/*.py` and warns loudly on
drift. Update `EXPECTED_MAE` in that script whenever this section changes.

`python src/train.py` still prints all-rows figures. Those exist only for the harness
fidelity check (`evaluate.check_fidelity`), which proves the harness measures what
training measures. They are not a claim about accuracy, and `FIDELITY_TARGETS` in
`src/evaluate.py` must be updated alongside this section whenever the pipeline changes.

## Hard Rules (never violate)

- No leakage: every feature for week W uses data strictly before W.
  Rolling features are `.shift(1)` then `.rolling(...)`, grouped by
  `player_id`. Applies to opponent/defense features too.
- No random/shuffled train-test split — walk-forward only (`validation.py`).
- Baseline reported next to every model result, same rows.
- Roll over a player's game sequence, not week number (bye weeks).
- Join on `player_id`, never player name.
- Verify column/API names before using — don't assume `nflreadpy` schema.

## Known Gotchas

- `nflreadpy` returns Polars — `.to_pandas()` once at load, never mix.
- Team codes differ by season for relocated franchises: OAK→LV, SD→LAC,
  STL→LA. Normalize before any join on team code (`features_context.py`).
- `spread_line`: negative = home team underdog (away favored). Verified
  against JAX @ GB wk1 2016.
- Efficiency ratios (e.g. yards-per-target) must be built from already-lagged
  rolling columns, not raw stats lagged after the fact.
- Use `np.nan`, not `pd.NA`, in ratio columns — `pd.NA` can produce `object`
  dtype, which XGBoost rejects.

## Maintenance

After completing a phase or a meaningful change, append findings to
LOG.md — what changed, new results, new gotchas found. Don't rewrite
or delete existing LOG.md entries, only append new ones.

If a change affects the locked-in results in the Verification section
above (new MAE/RMSE numbers, a new position added), update that section
to reflect the new state.

## Reference Docs

- `README.md` — the front door: headline result, the honest benchmark, quickstart
- `REPORT.md` — the technical writeup: method, validation protocol, full results,
  every negative result, limitations
- `LOG.md` — full phase-by-phase build history and findings. Note its preamble: MAE
  numbers rise mid-file because the pool convention changed in Phase 8
- `tests/` — the invariants. `test_features_lag.py` and `test_validation.py` are the
  leakage guards; `test_fidelity_slow.py` pins CLAUDE.md, the hook and the harness
  to the same numbers by importing the hook's `EXPECTED_MAE` rather than copying it
- Project spec (Claude Project knowledge) — feature/model/benchmark
  registries, extension recipes