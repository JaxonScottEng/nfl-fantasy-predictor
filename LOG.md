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

  ## Phase 5 — Iteration 1: Segmented Model
- Error analysis found model beats baseline by 11.1% MAE on high-volume players
  (baseline_pred >= 12, n=530) but is 1.7% WORSE than baseline on low-volume
  players (n=1899) -- aggregate 2.9% masked this split
- Built hybrid: use model prediction if baseline_pred >= 12, else fall back to baseline
- Result: MAE 4.677 (baseline) -> 4.543 (model) -> 4.492 (hybrid); 3.9% improvement
- Caveat: RMSE slightly worse for hybrid (6.360 vs model's 6.263) -- baseline
  fallback occasionally misses big on a breakout low-volume player. MAE improves,
  variance of large misses does not. Report both, don't cherry-pick MAE alone.
- Feature importances unchanged: baseline_last4 + targets_roll5 + receptions_roll5
  = ~63% of model's decisions -- usage dominates, consistent with domain expectation
  ## Phase 5 — Iteration 2 (negative result)
- Hypothesis: a games_played_this_season feature would help the model recognize
  thin-sample players and rely less on noisy rolling stats for them.
- Result: no meaningful change (Model MAE 4.543->4.545, Hybrid 4.492->4.491).
  Feature did not appear in top 15 importances.
- Conclusion: low-volume segment's error is not a signal-representation problem;
  appears to be a genuine data-sparsity/variance floor for these players.
  Reverted the feature -- kept codebase clean.
- Current best result: Hybrid MAE 4.491-4.492 (~4% improvement over baseline),
  concentrated entirely in high-volume players (~11% there vs ~-1.7% low-volume).

## Phase 6 — RB Added (second position)
- config.ACTIVE_POSITIONS is now ["WR", "RB"]; one SEPARATE model per position,
  never pooled (different usage semantics and scoring distributions).
- ARCHITECTURE FINDING: the "extend via config, not restructuring" assumption in
  CLAUDE.md did NOT fully hold. `features_opponent.py` summed fantasy points
  allowed across ALL of ACTIVE_POSITIONS into one number, so adding RB to config
  alone would have silently redefined WR's def_points_allowed_roll* features as
  "points allowed to WR **and** RB combined" -- changing WR's locked-in results.
  Fixed by grouping defense features per (defense_team, position) and adding
  `position` to the build_features join key. A defense soft against WRs is not
  necessarily soft against RBs, so this is the correct modeling unit anyway.
- New RB features (all built from already-lagged rolling columns, same pattern
  as yards_per_target): touches_roll{3,4,5} (carries + receptions),
  yards_per_carry_roll{3,4,5}, and rushing_first_downs_roll{3,4,5}.
  Motivation: RB mean 7.74 carries/game vs WR 0.18 -- the existing carries/
  rushing_* rollings capture volume but had no rushing-side efficiency ratio,
  no combined-opportunity count, and no down-role signal.
- Verified WR results are byte-identical after every step: MAE 4.677 / 4.543 /
  4.492 and RMSE 6.579 / 6.263 / 6.360, same 18 folds, same 2429 rows.

### RB results (2023 walk-forward, weeks 1-18, 18 folds, 1445 rows)
- Baseline (last-4 avg)  MAE 4.543, RMSE 6.445
- Model (XGBoost)        MAE 4.489, RMSE 6.112  (1.2% MAE improvement)
- Hybrid (segmented)     MAE 4.390, RMSE 6.243  (3.4% MAE improvement)
- Unlike WR, RB's RMSE improves for BOTH model and hybrid over baseline --
  the hybrid's RMSE penalty seen in WR Phase 5 does not reproduce here.
- Segment split mirrors WR's shape: high-volume (baseline_pred >= 12, n=348)
  model beats baseline by 9.0%; low-volume (n=1097) model is 3.5% WORSE.
  Same qualitative finding as WR, so the hybrid earns its place for RB too.
- RB hybrid threshold INHERITS WR's 12.0 rather than being separately tuned.
  Scanning for RB's best cutoff would be the post-hoc cherry-picking the WR
  analysis deliberately avoided. Revisit only off a dedicated RB error analysis.

### RB feature importances (final fold)
- baseline_last4 0.346, touches_roll3 0.139, touches_roll5 0.089,
  touches_roll4 0.042, rushing_yards_roll5 0.021, receiving_yards_roll5 0.021,
  targets_roll4 0.019, targets_roll5 0.018, implied_total 0.016
- The new `touches` feature is the #2/#3/#4 signal (~27% combined) -- clear
  validation that RB needed a combined-opportunity feature WR did not.
- Negative result worth recording: yards_per_carry_roll* and
  rushing_first_downs_roll* did NOT crack the top 15. RB prediction is driven
  by opportunity volume, not rushing efficiency -- consistent with the WR
  finding that usage dominates. Kept them (cheap, and they're the honest
  analogue set), but don't expect them to carry weight.

### New gotchas
- Expected nulls in the new RB features, both benign: 425 rows = a player's
  first career game (cold start, same as baseline_last4), and 950 rows where
  carries_roll* == 0 make yards_per_carry np.nan by design (pass-catching-only
  backs). XGBoost handles NaN natively. Hand-verified touches == carries +
  receptions on all 34,919 non-null rows (a naive all-rows equality check
  returns False purely because NaN != NaN -- not a bug).
- Regression hook (.claude/hooks/check_wr_regression.py) is now position-
  anchored: it matches "<POS> Baseline (last-4 avg) MAE:" per position. The
  earlier unanchored regex matched WR only by luck of print order.

