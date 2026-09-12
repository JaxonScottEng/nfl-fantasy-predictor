# features_context.py
import nflreadpy as nfl
import config

TEAM_CODE_FIXES = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    # add more here if other mismatches surface (e.g. Washington variants)
}

def normalize_team_codes(df, cols):
    df = df.copy()
    for col in cols:
        df[col] = df[col].replace(TEAM_CODE_FIXES)
    return df

def load_schedule_context():
    sched = nfl.load_schedules(config.SEASONS).to_pandas()

    keep = ["season", "week", "home_team", "away_team",
            "spread_line", "total_line"]
    sched = sched[keep].copy()

    # Normalize historical team codes (OAK->LV, SD->LAC, etc.) so this
    # table's team labels match the weekly-stats table's convention,
    # which already uses the current code across all seasons.
    sched = normalize_team_codes(sched, ["home_team", "away_team"])

    # Implied team total, home team's perspective.
    # Verified convention: spread_line NEGATIVE = home team is the
    # underdog (i.e. away team favored). Confirmed against JAX @ GB
    # week 1 2016 (GB favored ~3.5-4.5, spread_line was -3.5).
    sched["home_implied_total"] = (sched["total_line"] / 2) + (sched["spread_line"] / 2)
    sched["away_implied_total"] = (sched["total_line"] / 2) - (sched["spread_line"] / 2)

    return sched

if __name__ == "__main__":
    sched = load_schedule_context()
    print("shape:", sched.shape)
    print(sched.head(10))