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
- `streamlit run app.py` — launch the local GUI

Run every command from the PROJECT ROOT. `data_load.CACHE_PATH` is relative, so a
different cwd silently reads/writes a different cache.

## Verification

Always run after changes: `python src/train.py`

Confirm results are unchanged unless the change specifically targets them.
Each position is trained and evaluated separately (2023 walk-forward, weeks
1–18, 18 folds):

WR (2429 rows):
- Baseline MAE 4.677, RMSE 6.579
- Model MAE 4.543, RMSE 6.263
- Hybrid MAE 4.492, RMSE 6.360

RB (1445 rows):
- Baseline MAE 4.543, RMSE 6.445
- Model MAE 4.489, RMSE 6.112
- Hybrid MAE 4.390, RMSE 6.243

If any of these move without an intentional cause, stop and investigate before
proceeding — treat unexplained improvement as a leakage suspect, not a win.

These numbers are also enforced automatically: a `PostToolUse` hook
(`.claude/hooks/check_wr_regression.py`, registered in `.claude/settings.json`)
re-runs `src/train.py` after any edit to `src/*.py` and warns loudly on drift.
Update the expectations in that script whenever this section changes.

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