# NFL Fantasy Predictor

## Overview

Weekly PPR fantasy-points predictor using nflverse data. WR and RB, one model per
position, never pooled. Adding a position means: add it to `config.ACTIVE_POSITIONS`,
give it a feature list in `train.py`, and check that any feature aggregated across
positions (opponent defence) is grouped per position.

Python 3.13, virtualenv at `.venv`. nflreadpy returns Polars, converted to pandas at the
load boundary. pandas, scikit-learn, XGBoost, Streamlit.

## Project structure

| File | Purpose |
|---|---|
| `src/config.py` | Seasons, positions, rolling windows, train/validate/test split |
| `src/data_load.py` | nflverse pull and parquet cache |
| `src/features_*.py` | Usage, opportunity, snaps, expected points, opponent, betting lines |
| `src/build_features.py` | Joins the feature tables, asserts no row inflation |
| `src/validation.py` | Walk-forward fold generator |
| `src/train.py` | Fits the models; defines `FEATURE_COLS_BY_POSITION` and `XGB_PARAMS` |
| `src/registry.py` | id to function registry, used by metrics, pools and comparators |
| `src/metrics.py` | MAE, RMSE, R2, bias, Spearman, hit rate, weekly consistency |
| `src/pools.py` | Which players get scored. This is what makes results comparable |
| `src/comparators.py` | The methods being compared. Defines `SHIPPED_COMPARATOR_ID` |
| `src/model_quantile.py` | Quantile regression for the prediction range |
| `src/evaluate.py` | The harness, the fidelity check, and the CSV the app reads |
| `src/upcoming.py` | Predictions for weeks not yet played |
| `src/player_timeline.py` | A season week by week for the app |
| `app.py`, `gui/` | Streamlit app, views registered in one dict |
| `tests/` | Four files; `test_leakage.py` holds the rules the numbers depend on |

## Commands

```
python src/data_load.py                        pull and cache data
python src/evaluate.py [seasons...]            fidelity check and accuracy table
python src/evaluate.py --regression            machine-readable numbers (hook uses this)
python src/upcoming.py                         next week's projections
streamlit run app.py                           the app
pytest                                         62 tests, about 11 seconds
pytest -m slow                                 accuracy check against real data
python scripts/make_benchmark_chart.py         regenerate the README images
```

Run everything from the project root. Cache paths are relative, so a different working
directory silently reads and writes a different cache.

## Verification

Accuracy is measured on the top 40 WR/RB by projection per week, which is the group
published accuracy studies use. Scoring every rostered player instead makes MAE look about
30% better than it is. See `src/pools.py`.

Locked-in, 2023, top-40 pool. The hook checks these:

| | WR MAE | RB MAE |
|---|---|---|
| Baseline (last-4 avg) | 7.258 | 6.129 |
| Model (XGBoost) | 6.806 | 5.746 |
| Ensemble (shipped) | 6.759 | 5.758 |

Multi-season, 2021-2025, top-40 pool. Use these for any external claim:

| | WR MAE | RB MAE |
|---|---|---|
| Baseline | 7.047 | 6.464 |
| Model (XGBoost) | 6.514 | 6.073 |
| Ensemble (shipped) | 6.474 | 6.029 |

Published best-in-class on the same group: WR 4.84-4.94, RB 5.06-5.20. This project is
about 34% behind on WR and 19% behind on RB. Do not describe it as competitive with
commercial projections until those gaps close.

Single-season and multi-season results can disagree in direction. Judge changes on the
multi-season run. Eighteen weeks is too small a sample.

If a locked-in number moves without an intentional cause, stop and investigate. Treat an
unexplained improvement as a leakage suspect, not a win. A `PostToolUse` hook
(`.claude/hooks/check_wr_regression.py`) enforces this after any edit to `src/*.py`.

`python src/train.py` prints all-rows numbers. They exist only for
`evaluate.check_fidelity()` and are not an accuracy claim.

## The shipped projection

`comparators.SHIPPED_COMPARATOR_ID` is `ensemble_xgb_ridge`, the mean of XGBoost and
ridge. Everything user-facing reads that id. Its q10/q90 range is measured at 78% (WR) and
79% (RB) coverage against an 80% target.

A three-way ensemble including the quantile median scored better on MAE and was rejected.
A median sits below the mean on skewed scores, and the skew grows with volume, so the bias
lands on the best players: about 4.4 points for an elite receiver against 0.84 on average.

The hybrid is retired. It existed because the model used to lose to the baseline on
low-volume players, which stopped being true once expected-points features landed. Still
registered as a comparator so older comparisons stay reproducible.

## Hard rules

- No leakage. Every feature for week W uses data from before W. Rolling features are
  `.shift(1)` then `.rolling(...)`, grouped by `player_id`, including opponent features.
- Walk-forward only. No random or shuffled splits.
- The baseline is reported next to every model result, on the same rows.
- Roll over a player's game sequence, not week number, because of bye weeks.
- Join on `player_id`, never player name.
- Verify column and API names before using them. Do not assume the nflreadpy schema.

## Known gotchas

- nflreadpy returns Polars. Convert once at load, never mix.
- Team codes change for relocated franchises: OAK to LV, SD to LAC, STL to LA. Normalise
  before any join on team code.
- `spread_line` negative means the home team is the underdog.
- Efficiency ratios must be built from already-lagged rolling columns.
- Use `np.nan`, not `pd.NA`. `pd.NA` produces object dtype, which XGBoost rejects.
- Snap counts carry no `gsis_id`, only `pfr_player_id`. Bridge through weekly rosters.

## Maintenance

After a meaningful change, update REPORT.md: the version table in section 2.0, the results
in section 4.0, and section 3.0 if the change involved a judgement call worth recording.
Record features that were tried and removed, with their measured numbers.

If a change affects the locked-in results, update the Verification section above,
`EXPECTED_MAE` in the hook, and `FIDELITY_TARGETS` in `src/evaluate.py` together. They
must agree.

Keep the writing plain. The `no-ai-slop` skill in `.claude/skills/` lists what to avoid.

## Reference docs

- `README.md` — short front door: what it does, results, how to run it
- `REPORT.md` — the full writeup: version history, judgement calls, results, what was
  tried and removed, limitations
