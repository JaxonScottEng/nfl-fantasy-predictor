Phase 0: 12:54 2026-09-10 - Created virtual environment, downloaded all files, learned to use git and commits.

## Phase 1 — Baseline
- Cached WR weekly data, 2016–2024 (22,097 rows)
- Rolling last-4-game average baseline (lagged via shift(1), grouped by player_id)
- Verified by hand on Steve Smith week 9 (2016): matches manual calc (14.275)
- Result: MAE 4.907, RMSE 6.813 — this is the number to beat

## Phase 2 — Validation + Context
- Walk-forward fold generator built; verified all 22 folds (2023 season, incl. playoffs) have no leakage
- Weeks 19-22 are playoffs — small sample, may exclude from validation later
- Schedule/Vegas join: spread_line sign convention is NEGATIVE = home team is underdog
  (verified against JAX @ GB 2016 wk1, GB favored by 3.5 — matches)
- Implied totals: home = total/2 + spread/2, away = total/2 - spread/2

## Phase 3 — Feature Engineering
- Built lagged rolling usage features (targets, receptions, air yards, carries) and
  efficiency features (yards, TDs, yards-per-target), windows 3/4/5, verified via
  Steve Smith spot-check (values diverge correctly as history accumulates past
  the shortest window)
- Built lagged opponent-defense features (points allowed to position, rolling),
  hand-verified on ARI week 6 (32.025 = mean of weeks 2-5)
- Joined player + defense + schedule/Vegas context into one feature table (22097 rows, 51 cols)
- Bug found: team code mismatch (OAK vs LV, SD vs LAC — franchise relocations)
  caused 344 null implied_total rows; fixed by normalizing all team columns to
  current codes before joining. Confirmed 0 nulls after fix.
- Remaining nulls: 132 rows in def_points_allowed_roll4, all week 1 (expected
  cold-start — no prior game to roll from)
  
  ## Phase 4 — First Model
- XGBoost, per-fold retraining, walk-forward over 2023 regular season (weeks 1-18, playoffs excluded)
- Baseline (last-4 avg): MAE 4.677, RMSE 6.579
- Model (XGBoost, default-ish params): MAE 4.543, RMSE 6.263
- Improvement: 2.9% MAE reduction
- Fixed dtype bug: yards_per_target_roll* columns were 'object' dtype (pd.NA in
  division), rejected by XGBoost. Fixed with np.nan + explicit float cast.
- This is the CORE MILESTONE result — first honest model-beats-baseline comparison.
  Next: error analysis (Phase 5) before further tuning/features.