# NFL Fantasy Points Predictor

Predicts how many fantasy points an NFL wide receiver or running back will score in a
given week. Written in Python, with a Streamlit app for viewing the results.

## Background

Fantasy football players choose a lineup every week. To choose well you need an estimate
of how many points each player will score. Companies such as FantasyPros sell these
estimates. This project builds them from scratch and measures how close they get.

## Results

Accuracy is measured as mean absolute error (MAE): the average size of the miss, in
fantasy points. Lower is better. A MAE of 6.47 means a typical projection was about 6.5
points away from what the player actually scored.

| | Simple average of last 4 games | This model | Professional projections |
|---|---|---|---|
| Wide receiver | 7.05 | 6.47 | 4.84 - 4.94 |
| Running back | 6.46 | 6.03 | 5.06 - 5.20 |

![Accuracy comparison](docs/images/benchmark-gap.png)

The model beats a simple average by 8% for receivers and 7% for running backs. It does not
beat the professional services, which are about 34% and 19% more accurate.

Figures cover 2021 to 2025, 21,600 player-weeks. Professional figures come from Fantasy
Football Analytics and FantasyPros.

## Measuring fairly

Any accuracy number depends on which players you count. Scoring every receiver on a roster
includes around 100 bench players a week who score close to zero and are easy to predict.
That drags the average error down by about 30%.

This project reported 4.49 MAE for several weeks before the measuring tools were built.
The same model scored 6.47 once it was measured on the top 40 receivers per week, which is
the group professional accuracy studies use.

## Prediction ranges

Each projection comes with a range rather than a single number. The range is the model's
10th to 90th percentile estimate, so it widens for unpredictable players. Actual scores
landed inside the range 78% of the time for receivers and 79% for running backs, against a
target of 80%.

![One player's season](docs/images/player-season.png)

## The app

Three screens:

- Past Results: predictions against actual scores for any week.
- Upcoming Week: projections for the next unplayed week.
- Player Season: one player's whole season, actual scores plotted against the projection.

## How it was built

The code was written with Claude Code, an AI coding tool. Full detail is in
[REPORT.md](REPORT.md) section 5. The short version is that most of the effort went into
checks that catch mistakes:

- An automatic check runs after every code change and stops work if the accuracy numbers
  move without explanation, including when they improve.
- 62 tests cover the rules that keep the model honest, such as never letting a prediction
  see data from the week it is predicting.
- Features that did not work were measured, written down, and removed.

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt

python src/data_load.py      # download NFL data, about 25 MB, first run only
python src/evaluate.py       # accuracy tables
streamlit run app.py         # the app
```

Run commands from the project folder. Training takes a few minutes because the model is
refit once per week of the season.

```bash
pytest              # 62 tests, about 11 seconds
pytest -m slow      # accuracy check against real data, a few minutes
```

## Limits

- Covers wide receivers and running backs only.
- Projections more than one week ahead assume nothing changes, so they are a guide to
  current form rather than a forecast.
- The gap to professional services is mostly missing data. They have betting markets for
  individual players and injury information that is not publicly available.

## More detail

[REPORT.md](REPORT.md) covers the method, the results in full, and the things that did not
work.

## License

MIT. See [LICENSE](LICENSE).
