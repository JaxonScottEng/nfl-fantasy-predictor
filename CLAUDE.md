# NFL Fantasy Predictor

## Overview

Weekly PPR fantasy-points predictor using nflverse data. WR and RB currently,
one separate model per position — never pooled. Adding a position means: add it
to `config.ACTIVE_POSITIONS`, give it a feature list in `train.py`'s
`FEATURE_COLS_BY_POSITION`, and check that any position-aggregated feature
(e.g. opponent defense) is grouped per position rather than pooled.

## Stack

- Language: Python 3.11+, virtualenv (`.venv`)
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
- `src/build_features.py` — joins player + defense + context into one table
- `src/train.py` — walk-forward training: baseline vs model vs hybrid
- `src/registry.py` — generic id→function registry; the extension seam used by
  metrics, pools and comparators (and mirrored by `gui/prediction_ranges.py`)
- `src/metrics.py` — accuracy metrics registry (MAE, RMSE, R², bias, Spearman,
  top-N hit rate, CoV of weekly MAE). `train.py` imports `mae`/`rmse` from here
- `src/pools.py` — player-pool definitions; this is what makes results comparable
- `src/comparators.py` — anything scoreable (baseline, model, hybrid), one per function
- `src/evaluate.py` — the harness: `run_evaluation()` → row-level predictions,
  `summarize()` → metrics, `check_fidelity()` → proves it matches `train.py`
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

| | WR MAE | WR RMSE | RB MAE | RB RMSE |
|---|---|---|---|---|
| Baseline (last-4 avg) | 7.213 | 9.081 | 6.110 | 7.823 |
| Model (XGBoost) | 6.704 | 8.605 | 5.756 | 7.348 |
| Hybrid (segmented) | 6.688 | 8.629 | 5.814 | 7.495 |

**Multi-season, 2021–2025, top-40 pool** (180 folds, 21,600 scored rows — the
headline figure for any external claim):

| | WR MAE | RB MAE |
|---|---|---|
| Baseline | 7.045 | 6.492 |
| Model | 6.539 | 6.137 |
| Hybrid | 6.509 | 6.166 |

Run with `python src/evaluate.py 2021 2022 2023 2024 2025`.

### Where we actually stand

Published best-in-class on the same pool convention: **WR 4.84–4.94, RB 5.06–5.20**
(Fantasy Football Analytics, 11 seasons / 9 sources; FantasyPros recent). We are
roughly **34% behind on WR and 21% behind on RB**. Do not describe this project as
competitive with commercial projections until those gaps close.

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

`python src/train.py` still prints the older all-rows figures (WR 4.677/4.543/4.492,
RB 4.543/4.489/4.390). Those are kept only as the harness fidelity check
(`evaluate.check_fidelity`), which proves the harness measures what training measures.
They are not a claim about accuracy.

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

- `LOG.md` — full phase-by-phase build history and findings
- Project spec (Claude Project knowledge) — feature/model/benchmark
  registries, extension recipes