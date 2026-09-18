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


## Phase 7 — Streamlit GUI
- Local GUI so results are viewable without editing/re-running scripts by hand.
  Entry point `app.py` (thin: page config, sidebar, view dispatch), package `gui/`.
- Structured as a registry for extension: `app.py` holds `VIEWS = {name: render_fn}`,
  each view is its own file in `gui/views/` exposing `render(position, week, data)`.
  Adding a view = new file + one line in VIEWS.
- `gui/data_access.py` is the ONLY module that touches the pipeline; views never
  import train.py/build_features.py/upcoming.py directly. Positions come from
  `config.ACTIVE_POSITIONS`, so new positions appear with no GUI change.
- `gui/prediction_ranges.py` is a method registry: `get_prediction_range(row,
  method="mae_approx")` dispatches through `_METHODS`. Views build their method
  selector from `available_methods()`, so a new method appears in the UI with
  zero view edits. Only `mae_approx` exists now (point prediction +/- the hybrid's
  historical MAE for that position/volume segment), labeled in-app as an
  approximation with no coverage guarantee.
- `src/upcoming.py` (new, in src/ not gui/ -- it's pipeline logic, reusable
  outside the GUI and covered by the regression hook): predictions for a week
  that has NOT been played.

### How upcoming-week prediction stays leakage-free
- Append placeholder rows for the target week carrying only identity/schedule
  fields, with NaN for every raw stat and the target. Then run the EXISTING
  feature builders. Because they all lag via `.shift(1).rolling(...)` grouped by
  player_id, the placeholder row's features come strictly from earlier games, and
  its own NaN stats never feed its own features. Zero duplicated rolling logic.
- Same placeholder trick on the opponent-defense table so def_points_allowed_roll*
  is a properly lagged value rather than a missing join.
- The live model trains on all completed rows strictly before the target week --
  the same rule walk_forward_folds enforces, but with more history than the 2023
  validation model. So the validation MAE is an approximate guide to live error,
  not an exact one. Noted in the UI caption.

### Config change
- `config.SEASONS` extended 2016-2024 -> 2016-2026 so there IS an upcoming week
  (2026 wk1 complete; wk2 is next and already has Vegas lines).
- VERIFIED this does not move the locked-in numbers: walk_forward_folds trains on
  `season < VALIDATION_SEASON` and tests on VALIDATION_SEASON, so 2025-26 rows are
  never reached. All six MAE/RMSE values identical after the re-pull.
- Betting lines only exist for the near week (~88% of later unplayed games have
  null spread/total), which is fine -- only the next week is ever predicted.

### New gotchas
- `data_load.CACHE_PATH` is RELATIVE ("data/raw/..."). Running any pipeline entry
  point from a different cwd silently reads/writes a DIFFERENT cache. This bit
  during this phase: a pull run with cwd=src/ created `src/data/raw/weekly_stats.parquet`
  while the root cache stayed stale, and a verification run silently checked the
  OLD data. `gui/data_access.py` now does `os.chdir(PROJECT_ROOT)` at import to
  make the GUI immune. Always run pipeline scripts from the project root.
- `requirements.txt` was UTF-16 encoded (PowerShell redirect artifact); regenerated
  as UTF-8. Use `-Encoding utf8` if regenerating from PowerShell.
- Streamlit 1.64 deprecates `use_container_width`; use `width='stretch'`.
- Early-season predictions lean on last season's games (rolling windows span the
  season boundary by design, since features group by player_id only). Consistent
  with the existing baseline convention -- not a bug, but worth knowing when a
  week-2 projection looks driven by last year's form.


## Phase 8 — Evaluation harness, and the honest comparison

### Why
Every result so far was measured only against our own last-4-average baseline, so
"4.492 MAE" said nothing about whether this is competitive with anything. Built a
measurement layer first, then looked.

### The registry convention now actually exists
CLAUDE.md claimed a "registry + recipe" extension pattern; the only thing implementing
it was gui/prediction_ranges.py's private _METHODS dict. `src/registry.py` generalizes
it, and metrics / pools / comparators all extend by writing one decorated function:
- `src/metrics.py` — mae, rmse, r2, mean_error (bias), spearman_within,
  top_n_hit_rate, cov_weekly_mae. train.py now imports mae/rmse from here instead
  of defining its own (verified byte-identical output).
- `src/pools.py` — all / top_n_by_projection / union_top_n.
- `src/comparators.py` — baseline_last4, xgb_model, hybrid. Shares `XGB_PARAMS`
  with train.py so a comparator can never silently drift from what training fits.
- `src/evaluate.py` — `run_evaluation()` returns ROW-LEVEL predictions, not
  aggregated metrics, because pooled MAE across folds is NOT the mean of per-fold
  MAEs (folds differ in size) and CoV-of-weekly-MAE needs the per-week breakdown.

### Fidelity gate passed
Harness at pool=all, seasons=[2023] reproduces train.py exactly: WR
4.677/4.543/4.492, RB 4.543/4.489/4.390. Kept as `evaluate.check_fidelity()`.

### PLAN DEVIATION (for the better)
The plan called for changing `walk_forward_folds` to yield a 4-tuple with the season.
Unnecessary: the harness loops over seasons and passes `validation_season` itself, so
it already knows the season. Left the 3-tuple alone — no breaking change, no broken
intermediate state, and no risk to the locked-in numbers.

### THE HEADLINE FINDING: we are well behind the field
Published studies (Fantasy Football Analytics, 11 seasons / 9 sources; FantasyPros)
evaluate the **top 40 WR/RB by projected points per week**, not every player. Our
2429 WR rows/season was ~140 WRs/week — 3.5x that pool — and the extra WR4/WR5s
score near zero and are trivially predictable, which dragged MAE down by ~30%.

Top-40 pool, 2021-2025, 180 folds, 21,600 scored rows:

| | WR MAE | RB MAE |
|---|---|---|
| Baseline | 7.045 | 6.492 |
| Model | 6.539 | 6.137 |
| Hybrid | 6.509 | 6.166 |
| **Published best-in-class** | **4.84-4.94** | **5.06-5.20** |

So: ~34% behind on WR, ~21% behind on RB. The old 4.492 was an artifact of the pool,
not a competitive result. Per-season MAE is stable (WR model 6.305-6.704 across
2021-2025), so this is a real gap, not noise.

### Diagnostics the new metrics immediately surfaced
- `mean_error` is negative for the model (-0.135 WR, -0.184 RB pooled; -0.76/-1.03 on
  2023 alone) — it systematically UNDER-projects top-40 players. Classic
  regression-to-the-mean shrinkage from training on a pool dominated by low-volume
  players. A concrete, fixable target.
- The baseline's R^2 on the top-40 pool is NEGATIVE (-0.043 WR) — worse than
  predicting the weekly mean. It only looked respectable because deep-bench rows
  flattered it.
- **RB's hybrid is now WORSE than the plain model** (6.166 vs 6.137). The 12.0
  threshold was tuned against the all-rows metric; on the pool that matters it hurts.
  Reinforces retiring the hybrid in favour of a calibrated p_play (Stage 4).
- top_n_hit_rate (top 12 of the top 40) is only 0.42 WR / 0.46 RB.

### Official metric switched
CLAUDE.md's Verification section is now the top-40 pool, and
`.claude/hooks/check_wr_regression.py` runs `python src/evaluate.py --regression`
(single season, ~same cost as before) instead of parsing train.py. Locked-in 2023
top-40: WR 7.213/6.704/6.688, RB 6.110/5.756/5.814. The all-rows numbers survive only
as the fidelity check, explicitly not as an accuracy claim.


## Phase 9 — Playoff contamination fix

### The bug
`season_type` was unused. The cache holds 7,879 POST rows, and
`MAX_VALIDATION_WEEK = 18` filtered playoffs ONLY for the validation season — so
2016-2022 playoff games sat inside training data AND inside every rolling window.
Playoff games have a different player pool (14 teams, resting starters), so this
mixed unlike things into both training and each player's "last 4 games".

### The fix
`data_load.filter_regular_season(df)` applied in all four modules that read raw
weekly data — `build_target.get_target_table`, `features_player.build_player_features`,
`features_opponent.build_opponent_defense_features`, and
`upcoming.build_upcoming_feature_table`. It runs BEFORE any rolling; filtering only
the prediction targets would still leave playoff games inside the windows.

Deliberately called explicitly in four places rather than hidden inside
`load_weekly()`, because forgetting it in one place is precisely the bug being fixed —
and `upcoming.py` needed it too, so a live projection's rolling window means the same
thing as a training row's.

Verified: 7,879 rows dropped, and max week per season is now exactly 17 for <=2020 and
18 for >=2021 — zero postseason weeks anywhere in the feature table.

### Effect on results (all slightly WORSE, which is correct)
Removing playoff games removes training data and shortens some rolling windows.

Top-40 pool, 2021-2025:

| | WR before | WR after | RB before | RB after |
|---|---|---|---|---|
| Baseline | 7.045 | 7.100 | 6.492 | 6.548 |
| Model | 6.539 | 6.578 | 6.137 | 6.168 |
| Hybrid | 6.509 | 6.550 | 6.166 | 6.196 |

All-rows (fidelity check only): WR 4.677/4.543/4.492 -> 4.685/4.559/4.502;
RB 4.543/4.489/4.390 -> 4.551/4.496/4.393.

A worse number here is the correct outcome, not a regression — the previous figure was
partly borrowed from data that shouldn't have been in scope. Three places had to be
updated together: CLAUDE.md's Verification section, `EXPECTED_MAE` in the hook, and
`FIDELITY_TARGETS` in `src/evaluate.py`.

### Note on the hook working as designed
Each of the four edits tripped the regression hook with a loud failure before the
expectations were updated. That is the intended behaviour for a change that moves the
frozen numbers — the alarm fired every time, and the numbers stabilised at
WR 7.286/6.760/6.760 and RB 6.155/5.785/5.838 (2023 top-40) once all four readers were
filtered, which is itself evidence the four call sites were the complete set.


## Phase 10 — Opportunity, snap share and expected points

### What was added
- `features_player.OPPORTUNITY_COLS` — target_share, air_yards_share, receiving_epa,
  rushing_epa, receiving_first_downs, receiving_yards_after_catch. All were already
  sitting unused in the cached parquet. Excluded on purpose: `wopr` (exactly
  1.5*target_share + 0.7*air_yards_share, so the primitives carry it), `racr`
  (explodes near zero air yards), `pacr` (99%+ null off QB).
- `src/features_snap.py` — offensive snap share. load_snap_counts has only
  `pfr_player_id`, so it bridges through load_rosters_weekly's pfr_id (name joins
  are forbidden). Unmatched rows stay NaN, never 0 — 0 would assert "played no
  snaps" when the truth is "unknown". Final null rate 12.6% WR / 14.3% RB.
- `src/features_expected.py` — nflverse ffopportunity expected points. Joins 100%
  on player_id. `season` arrives as String and `week` as float; both need casting.
- `ROLLING_INPUT_COLS` consolidates USAGE+PROD+OPPORTUNITY so upcoming.py cannot
  drift from the training pipeline's input list.
- `build_features._merge_player_week` asserts row count is unchanged after each
  join — a duplicated key on the right silently multiplies rows and would corrupt
  every metric without raising.

### PLAN DEVIATION: no same-week expected-points comparator
The plan called for a `nflverse_exp` comparator using same-week
`total_fantasy_points_exp`. Dropped: expected points for week W are derived from
week W's plays, so that comparator would be an ORACLE with access to the outcome's
own inputs, not a peer projection — and publishing it next to real projections
would be misleading. Only lagged rolling values are produced, so the `_current`
column and its guard assertion were never needed.

### Results (top-40 pool, 2021-2025)

| | before | after | vs field |
|---|---|---|---|
| WR model | 6.578 | **6.514** | 4.84-4.94 |
| RB model | 6.168 | **6.073** | 5.06-5.20 |

Gap narrowed from ~34%/21% to ~32%/17%. Also: WR mean_error -0.172 -> +0.010
(systematic under-projection of top players essentially eliminated), RB Spearman
0.363 -> 0.382, both CoV improved.

**Expected points is the single strongest signal in the model.**
`total_fantasy_points_exp_roll5` alone carries 40.8% of WR importance; the whole new
block is 79.2% of WR and 66.0% of RB importance. It works because it measures what a
player's opportunities were WORTH, which is far more stable than what he actually
scored — touchdown luck is the biggest source of weekly noise.

### IMPORTANT: single-season and multi-season disagreed in direction
On 2023 alone WR got slightly WORSE (6.760 -> 6.806) while the 5-season average got
better (6.578 -> 6.514). One season of 18 folds is too noisy to judge a change on.
Always evaluate on the multi-season run; the hook's single-season numbers are a
regression tripwire, not evidence.

### THE HYBRID IS NOW OBSOLETE
Its entire justification was that the model LOST to the baseline on low-volume
players. After these features, the model wins in both segments:

| low-volume segment | before | after |
|---|---|---|
| WR | -1.7% (worse) | **+4.0% better** |
| RB | -3.5% (worse) | **+0.9% better** |

On all rows the plain model now beats the hybrid (WR 4.384 vs 4.506), so the
baseline fallback is pure drag. This also means Stage 4's hurdle model is no longer
needed for the reason it was planned — the low-volume problem is already solved.

### Gap found by the GUI test
`upcoming.py` builds its own feature table and broke on the new columns
(KeyError on offense_pct_roll3 etc). Fixed by giving `build_snap_features` and
`build_expected_points_features` an `extra_keys` parameter that appends NaN
placeholder rows — the same trick upcoming.py already used for player features, so
an unplayed week gets a properly lagged value instead of a missing join. Caught only
because the GUI is exercised via streamlit.testing AppTest; worth keeping that habit.


## Phase 11 — Availability features (NEGATIVE RESULT, reverted) + hybrid retired

### Hypothesis
The model had no idea whether a player was injured, so injury-report status, practice
participation, teammate absence and time-since-last-game should help — particularly on
the top-40 pool, where a player who is ruled out would be projected highly and score 0.

### Why the premise was wrong (verified, not assumed)
**A player listed "Out" has no weekly stats row at all** — 996 of 996 "Out" player-weeks
in 2023 have no row in load_player_stats. Inactive players are simply ABSENT from the
data, so they are never predicted and the "projected 8, scored 0" error the feature was
meant to catch does not exist in this dataset. What remained was thin:
- only 5% of rows carry any injury designation (1,943 Questionable, 3 Doubtful, 0 Out)
- practice_status is 82.6% null (present only for players already on the report)
- 18% of rows do have a same-position teammate listed Out, which was the best hope

### Both variants measured, both null (top-40 pool, 2021-2025)

| | Stage 3 baseline | + counts | + vacated-points weighting |
|---|---|---|---|
| WR model MAE | 6.514 | 6.519 | 6.511 |
| RB model MAE | 6.073 | 6.109 | 6.088 |

The second variant weighted teammate absence by the absent player's most recent expected
points (via merge_asof), since a WR1 sitting vacates far more opportunity than a WR5 and
a plain count treats them identically. It was better than the count but still null on MAE.
WR ranking metrics improved slightly (Spearman 0.252 -> 0.267, top-12 hit rate
0.419 -> 0.435) while RB's got marginally worse — within noise across 180 folds. No
availability feature reached the top 12 importances for either position.

### Reverted, per the Phase 5 iteration-2 precedent
`src/features_availability.py` deleted, join and feature-list entries removed. Confirmed
the revert restores the committed numbers exactly (WR 7.258/6.806/6.821,
RB 6.129/5.746/5.784).

**The most valuable consequence of reverting: CLAUDE.md's leakage rule did NOT have to be
amended.** The plan called for relaxing "strictly before week W" to "knowable at kickoff"
so pre-kickoff injury reports would be legal. Since the features don't help, the project
keeps its strictest invariant intact. Loosening a core safety rule to buy nothing would
have been a bad trade.

Worth knowing if this is revisited: the signal is absent because of how nflverse shapes
the data, not because injuries don't matter. A dataset with a row per rostered player per
week (including inactives) would make availability modelling essential — and that is also
what a hurdle model would need to be worth building.

### Hybrid retired as the shipped prediction
Its premise died in Phase 10: the model now beats the baseline in both volume segments, so
the baseline fallback is pure drag (all-rows WR 4.384 model vs 4.506 hybrid). `model_pred`
is now what upcoming.py sorts by and what the GUI displays; `gui/prediction_ranges` and
`data_access.get_segment_mae` key off `model_error` instead of `hybrid_error`.

`hybrid` stays registered in comparators.py so every comparison in Phases 5-10 remains
reproducible. Ranges are still segmented by volume even though the router is gone, because
error scales with scoring level (~6.9 MAE for high-volume WRs vs ~3.7 for low) and a single
band would be far too wide at the bottom and too narrow at the top.


## Phase 12 — Quantile regression and ensembling

### The objective-alignment insight
Our headline metric is MAE, which is minimized by the conditional MEDIAN, but the model
trained on `reg:squarederror`, which optimizes the conditional MEAN. Training at
alpha=0.5 therefore optimizes the thing we actually report. It worked: xgb_q50 beat
xgb_model on MAE (WR 6.499 vs 6.514, RB 6.020 vs 6.073).

But q50 is NOT strictly better, and this is the important part: its bias is -2.08 (RB)
and -2.26 (WR), and its R^2 collapses (WR 0.002 vs 0.073). A median sits below the mean
on a right-skewed distribution. It trades calibration for MAE.

### Surprise: ridge alone beats XGBoost
WR 6.485 vs 6.514, RB 6.052 vs 6.073. A plain linear model over the same features beats
the tuned tree model on both positions. Worth remembering before assuming the tree model
is near its ceiling -- and it is exactly why an ensemble of the two helps: they err
differently.

### Ensembling reproduces the literature's most robust finding
Aggregation beat every individual member. Top-40 pool, 2021-2025:

| | WR MAE | RB MAE | bias | WR R^2 |
|---|---|---|---|---|
| ensemble xgb+q50+ridge | **6.427** | **5.973** | -0.84 | 0.073 |
| **ensemble xgb+ridge (SHIPPED)** | 6.474 | 6.029 | **-0.13** | **0.083** |
| ridge | 6.485 | 6.052 | -0.27 | 0.076 |
| xgb_q50 | 6.499 | 6.020 | -2.26 | 0.002 |
| xgb_model (previous shipped) | 6.514 | 6.073 | +0.01 | 0.073 |

### We shipped the SECOND-best MAE on purpose
The 3-way won MAE but its average bias of -0.84 hides the real problem: the median's
downward pull scales with skew, and skew scales with volume. Checked on a real player --
Puka Nacua, 2026 wk2: xgb_model 22.7, 3-way ensemble **18.4**. About -4.4 on an elite
WR, which would read visibly wrong beside any commercial projection, for a 0.7% MAE gain.

Lesson worth keeping: an aggregate bias figure can hide a large, structured error
concentrated exactly where it matters most. Check a metric on real individual rows
before trusting its average.

### Intervals are real now, and the old band was lying
q10-q90 coverage measured **0.778 WR / 0.785 RB** on the top-40 pool (0.825 / 0.821 on
all rows) against an 0.80 target -- honest within a couple of points.
They are also ~19 points wide, where the retired +/- MAE band was ~13.8 and flat for
everyone in a segment. So that band was not merely approximate, it was UNDERSTATING
uncertainty, and it never had a coverage guarantee to check. `gui/prediction_ranges`
now offers "quantile" first; `mae_approx` stays for comparison.

### Architecture: the evaluation layer now owns predictions_detail.csv
train.py cannot import comparators.py (comparators imports train for the feature lists
-- a cycle), so it could never produce an ensemble prediction without duplicating the
definition. Moved the writer to `evaluate.write_prediction_details()`, which reaches the
registry, so the shipped ensemble and its interval are defined in exactly one place.
train.py stays the fidelity anchor and no longer writes the file.

## Phase 13 — Player season view

New GUI view: one player's whole season, weeks across the bottom, PPR points up the
side, actual scores as points and the projection as a line with its q10-q90 band. Added
as `gui/views/player_season.py` plus one line in `app.py`'s VIEWS -- the registry
extension point working exactly as designed in Phase 7.

`src/player_timeline.py` stitches two DIFFERENT machines and labels the seam:
- PLAYED weeks come from the walk-forward harness -- genuine out-of-sample predictions,
  the same ones every accuracy number here is computed from.
- UNPLAYED weeks come from `upcoming.predict_weeks`, one fit covering all of them.

### Why future weeks are an outlook, not a projection
For any future week the lagged rolling features are IDENTICAL, because no games are
played in between -- week 10's features are week 2's features. Only opponent, that
defense's form and the market line differ. On top of that ~88% of later unplayed games
have no spread/total posted (verified: 2026 weeks 2-3 have lines, weeks 4 and 10 are
100% null). So one model fit serves every remaining week, and the view labels those
weeks as a current-form outlook. They do still vary by opponent (Nacua: 19.5 vs DEN,
21.7 vs PHI), so the line is not flat -- just not genuinely game-specific.

### Two real bugs found by building this
- **Cache skew made a week disappear.** `season_week_status` originally read played/
  unplayed from the SCHEDULE, but the schedule cache and the weekly-stats cache refresh
  independently. With schedules newer, week 2 was "played" with no player rows -- so it
  was neither scored nor projected and silently vanished from the chart. Fixed by
  deriving "played" from the player data, so a week we cannot score is projected
  instead, which is visible.
- **Schedules were fetched over the network on every run** (the only table not cached),
  which made the live path depend on connectivity -- and it failed mid-build with a DNS
  error. Now parquet-cached via `features_context.load_schedules_cached()`, with
  "Refresh Data" refreshing it, since that table carries the current season's results
  and newly posted lines.
