# features_context.py
import nflreadpy as nfl
import config

def load_schedule_context():
    sched = nfl.load_schedules(config.SEASONS).to_pandas()

    # Inspect columns before trusting names — nflreadpy is new.
    # Expect something like: season, week, home_team, away_team,
    # spread_line, total_line

    keep = ["season", "week", "home_team", "away_team",
            "spread_line", "total_line"]
    sched = sched[keep].copy()

    # Implied team total: derived from the over/under and the spread.
    # Convention: spread_line is from the home team's perspective
    # (negative = home favored). Verify this against a known game.
    sched["home_implied_total"] = (sched["total_line"] / 2) + (sched["spread_line"] / 2)
    sched["away_implied_total"] = (sched["total_line"] / 2) - (sched["spread_line"] / 2)

    return sched

if __name__ == "__main__":
    sched = load_schedule_context()
    print("shape:", sched.shape)
    print(sched.head(10))