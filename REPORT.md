# Technical report

A weekly fantasy-football points projection system for NFL wide receivers and running
backs, built to a measurement standard rather than to a leaderboard. This report assumes
you have read the [README](README.md); it does not re-introduce the project.

---

## 1. Objective and success criterion

The goal was never "build a model that predicts fantasy points" — that is a two-hour
exercise. It was to build a system whose accuracy claim survives contact with an outside
reference.

That distinction drove the success criterion: **mean absolute error, in PPR points, on a
declared player pool, compared against published commercial accuracy figures.** Every part
of that sentence is load-bearing:

- **MAE** because lineup and DFS decisions care about points, and because it is the metric
  published accuracy studies report.
- **On a declared pool** because MAE is meaningless without one. See §6.1 — this is the
  single most important finding in the project.
- **Against published figures** because "beats my own baseline" is a claim about the
  baseline, not about the model.

Secondary criteria: calibrated prediction intervals (measured coverage, not asserted), and
rank correlation within week, since start/sit is an ordering problem.

## 2. Data

| | |
|---|---|
| Source | [nflverse](https://github.com/nflverse) via `nflreadpy` |
| Seasons | 2016–2026 (2026 partial) |
| Grain | one row per player per game |
| Raw rows | 183,373 |
| Modelled rows | 38,734 (WR + RB, regular season only) |
| Scored rows | 21,600 (top-40 pool, 2021–2025) |

Six tables are joined: weekly player stats, schedules with betting lines, opponent defense
aggregates, snap counts, nflverse's expected-points model (`ff_opportunity`), and injury
reports (later removed — §6.2). All are cached to local parquet; only an explicit refresh
re-downloads.

Two data traps worth recording:

- **Team codes are not stable across seasons.** Franchise relocations (OAK→LV, SD→LAC,
  STL→LA) silently broke 344 joins before being normalised in `features_context.py`.
- **Snap counts carry no `gsis_id`**, only a Pro-Football-Reference id, so they cannot be
  joined to player stats directly. They are bridged through weekly rosters, which carry
  both (92% match). Joining on player *name* would have been easy and wrong — names are
  not unique and change spelling.

## 3. Method

**One model per position, never pooled.** A WR's carries and an RB's carries mean different
things, and pooling lets one position's sample size distort the other's fit. WR uses 64
features, RB 67, with separate feature lists in `train.py`.

**Feature families** (all rolling, all lagged, windows of 3/4/5 games):

| Family | Examples |
|---|---|
| Usage | targets, receptions, carries, air yards, touches |
| Production | receiving/rushing yards, TDs, first downs, YAC |
| Opportunity share | target share, air-yards share — team-normalised |
| Efficiency | yards per target, yards per carry, receiving/rushing EPA |
| Expected production | nflverse expected fantasy points, expected receptions, expected TDs |
| Snap share | offensive snap percentage |
| Opponent | points allowed to that position, per defense, per position |
| Market | spread, total, and the derived implied team total |

**Shipped model:** an equal-weight mean of XGBoost and ridge regression
(`comparators.SHIPPED_COMPARATOR_ID`). A separate quantile model
(`objective="reg:quantileerror"`, α = 0.1/0.5/0.9) supplies the prediction interval.

The single strongest feature is nflverse's **expected fantasy points**:
`total_fantasy_points_exp_roll5` alone accounts for **40.8%** of WR feature importance, and
the expected/opportunity block accounts for 79.2% (WR) and 66.0% (RB). It works because it
measures what a player's opportunities were *worth* rather than what he happened to score
— touchdown incidence is the noisiest component of weekly fantasy scoring.

## 4. Validation protocol

This is the part of the project I would defend hardest.

1. **Walk-forward only.** For each week W of a season, train on every completed game
   strictly before W and predict W. One model fit per fold, 180 folds across 2021–2025. No
   random splits anywhere — a shuffled split on time-series sports data leaks the future
   into the past and inflates every metric.
2. **Every feature is lagged at source.** All rolling features are `.shift(1)` *then*
   `.rolling(...)`, grouped by `player_id`. Shifting after rolling would include week W's
   own value in its own predictor. `tests/test_features_lag.py` pins this with a case where
   the two orderings give different answers.
3. **Ratios are built from already-lagged columns**, never from raw stats lagged afterwards.
4. **Rolling windows follow a player's game sequence, not week number**, so bye weeks and
   missed games do not silently shift a window.
5. **Regular season only**, filtered *before* any rolling — filtering only the prediction
   targets would still leave playoff games inside the windows (§6.3).
6. **The baseline is scored on identical rows** to every model, in the same fold, from the
   same pool. Comparing against a baseline measured elsewhere is how projects accidentally
   beat nothing.

**Projecting unplayed weeks** needs care, and `upcoming.py` handles it with a placeholder
trick: append rows for the target week carrying only identity and schedule fields, with NaN
for every statistic, then run the *existing* lagged feature builders. Because they all shift
before rolling, the placeholder row's features derive strictly from earlier games, and its
own NaNs never feed itself. No rolling logic is duplicated, so the live path cannot drift
from the training path.

## 5. Results

All figures: top-40-by-projection pool, 2021–2025, 180 walk-forward folds, 21,600 scored
rows. Lower MAE is better.

| Comparator | WR MAE | RB MAE | WR bias | WR R² |
|---|---|---|---|---|
| Naive baseline (last-4 average) | 7.047 | 6.464 | +1.14 | −0.06 |
| Expected points (last-4 average) | 6.775 | 6.251 | +0.50 | 0.01 |
| Retired hybrid | 6.503 | 6.115 | −0.53 | 0.06 |
| XGBoost alone | 6.514 | 6.073 | +0.01 | 0.07 |
| Ridge alone | 6.485 | 6.052 | −0.27 | 0.08 |
| Quantile median (q50) | 6.499 | 6.020 | −2.26 | 0.00 |
| **Shipped: XGBoost + ridge** | **6.474** | **6.029** | **−0.13** | **0.08** |
| *3-way incl. q50 (rejected, §7)* | *6.427* | *5.973* | *−0.84* | *0.07* |
| **Published best-in-class** | **4.84–4.94** | **5.06–5.20** | — | — |

**Per-season stability** (shipped model, MAE):

| | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|
| WR | 6.412 | 6.353 | 6.759 | 6.497 | 6.351 |
| RB | 6.332 | 6.148 | 5.758 | 5.823 | 6.085 |

The spread is narrow enough that the gap to published figures is a real gap, not noise.

**Honest summary:** 8% (WR) and 7% (RB) better than a naive baseline; 34% (WR) and 19% (RB)
*worse* than commercial projections. Rank correlation within week is 0.27 (WR) and 0.39 (RB).

A caution the data itself taught: **single-season and multi-season results can disagree in
direction.** Adding the opportunity feature family made WR *worse* on 2023 alone (6.760 →
6.806) and *better* across five seasons (6.578 → 6.514). One season of 18 folds is too noisy
to judge a change on — a lesson that would have saved two wrong conclusions had it been
learned earlier.

## 6. Negative results

These are recorded because they were the most instructive part of the project, and because
a log of only successes is evidence of poor record-keeping rather than good luck.

### 6.1 The headline number was a measurement artifact

The project reported **4.49 MAE** for months. Building the evaluation harness revealed that
published studies score only the **top 40 WR by projection each week**, while this project
was scoring every WR with a prior game — about 140 per week. The extra 100 are deep-bench
players who score near zero and are trivially predictable, and they dragged MAE down by
roughly 30%.

The same model, measured on the comparable pool, scores **6.47**. Nothing about the model
changed; the measurement was wrong. This is the finding the whole project rests on, and it
made the headline number worse.

### 6.2 Injury and availability features: no effect, reverted

**Hypothesis:** the model has no idea whether a player is injured, so injury report status,
practice participation, teammate absence and time-since-last-game should help — especially
on the top-40 pool, where a player who is ruled out would be projected highly and score 0.

**Why the premise was false:** a player listed "Out" **has no weekly stats row at all** —
verified at 996 of 996 "Out" player-weeks in 2023. Inactive players are absent from the
data entirely, so they are never predicted, and the error the feature targeted cannot occur.

**Measured** (top-40, 2021–2025):

| | Before | Teammate counts | Vacated-opportunity weighting |
|---|---|---|---|
| WR | 6.514 | 6.519 | 6.511 |
| RB | 6.073 | 6.109 | 6.088 |

Both variants null. The second weighted teammate absence by the absent player's recent
expected points, since a WR1 sitting vacates far more opportunity than a WR5 — better than
a raw count, still null. **Reverted.**

The most valuable consequence of reverting: the project's leakage rule did **not** have to
be relaxed. The plan had called for amending "strictly before week W" to "knowable at
kickoff" so pre-kickoff injury reports would be legal. Since the features bought nothing,
the strictest invariant stayed intact.

### 6.3 Playoff contamination

`season_type` was unused, and the week filter applied only to the validation season — so
2016–2022 playoff games sat inside training data *and* inside rolling windows. Playoff games
have a different player pool (14 teams, rested starters). Fixing it made results slightly
**worse** (WR 6.539 → 6.578), which is correct: the earlier figure was partly borrowed from
data that should not have been in scope.

### 6.4 The hybrid, retired

For several phases the shipped prediction was a "hybrid" that used the model for
high-volume players and fell back to the baseline for low-volume ones, because the model
genuinely lost to the baseline on the latter. After the expected-points features landed,
that stopped being true:

| Low-volume segment | Before | After |
|---|---|---|
| WR | −1.7% (worse than baseline) | **+4.0% better** |
| RB | −3.5% (worse than baseline) | **+0.9% better** |

The hybrid's entire premise had dissolved, so it was retired. It survives as a comparator
so that earlier phases remain reproducible.

## 7. Shipping the second-best MAE on purpose

A three-way ensemble adding the quantile median won on the declared metric — WR 6.427 vs
6.474, RB 5.973 vs 6.029 — and was **rejected**.

A median sits below the mean on a right-skewed distribution, and fantasy scoring skew grows
with volume. So the bias concentrates on exactly the players that matter. Checked on a real
player rather than trusting the aggregate:

| Puka Nacua, 2026 week 2 | |
|---|---|
| XGBoost | 22.7 |
| 3-way ensemble (incl. q50) | **18.4** |

Roughly **−4.4 points on an elite WR**, hidden inside an average bias of −0.84. A projection
system that under-sells the best players by four points reads as broken next to any
commercial source, and would have bought a 0.7% MAE improvement for it.

The transferable lesson: **an aggregate metric can conceal a large, structured error
concentrated where it matters most.** Check on individual rows before trusting an average.

## 8. Calibration

The prediction interval is the model's own 10th–90th percentile, learned per player, so it
widens for volatile players rather than applying one flat band.

| Pool | WR coverage | RB coverage | Target |
|---|---|---|---|
| Top-40 | 0.778 | 0.785 | 0.80 |
| All rows | 0.825 | 0.821 | 0.80 |

Honest within a couple of points in both directions. It also replaced a band that was
*lying*: the previous "± historical MAE" interval was ~13.8 points wide where ~19 is needed
for 80% coverage, so it was understating uncertainty rather than merely approximating it.

## 9. Limitations

- **Two positions only.** QB and TE are straightforward extensions (the per-position
  architecture already exists) but need their own thresholds and feature lists. Kickers and
  team defenses need an entirely new target — `fantasy_points_ppr` is 0.00 for all 5,305
  kicker rows.
- **The test season has never been touched.** 2024 is reserved in config and has not been
  used as a final holdout; all reported figures are validation-season walk-forward.
- **Working-directory coupling.** Scripts in `src/` resolve cache paths relative to the
  current directory, so running them from elsewhere silently reads a different cache. The
  GUI defends itself; the scripts do not. Documented rather than fixed, because the fix
  touches every file under the regression hook for no reviewer-visible gain.
- **Multi-week projections are a form outlook, not a forecast.** Beyond the next game, lagged
  features cannot advance (no games have been played) and ~88% of later fixtures have no
  betting line posted, so those weeks vary only by opponent.
- **The remaining gap is mostly a data gap.** The two clearest missing inputs — player prop
  markets and true availability signal — are commercial data problems, not modelling ones.

## 10. What I would do differently

1. **Declare the player pool in week one.** The single largest correction in the project was
   a measurement definition, not a model change. It should have been the first decision.
2. **Write the evaluation harness before the third feature family.** Every feature added
   before it was judged against a number that turned out to be incomparable.
3. **Evaluate on multiple seasons from the start.** Two conclusions were nearly drawn from
   single-season movements that reversed across five.
4. **Check aggregate metrics on individual rows earlier.** The −4.4 elite-player bias was
   invisible in every summary table that had been produced up to that point.

---

## Appendix: reproducing the numbers

```bash
pytest                                   # fast invariant suite (~11s)
pytest -m slow                           # fidelity gate against real data
python src/evaluate.py                   # fidelity check + top-40 table, 2023
python src/evaluate.py 2021 2022 2023 2024 2025   # the headline multi-season table
python scripts/make_benchmark_chart.py   # regenerate the comparison chart
```

Run everything from the project root. The first run downloads roughly 25 MB from nflverse;
a full multi-season evaluation takes several minutes because it refits per fold by design.
