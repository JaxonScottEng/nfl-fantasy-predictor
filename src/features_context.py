# features_context.py
import nflreadpy as nfl
import config
from data_load import cached_table

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

def load_schedules_cached(force_refresh=False):
    """
    Schedules, results and betting lines, cached to parquet like every other table.

    Previously this was the one nflverse call made on every run, which made the live
    projection path depend on the network each time -- and it did fail once mid-build.
    Note the tradeoff: the current season's rows change weekly as games finish and
    lines post, so this cache must be refreshed to see a new week. "Refresh Data" in
    the GUI does that, matching the project's explicit-refresh-only convention.
    """
    return cached_table(
        "schedules",
        lambda: nfl.load_schedules(config.SEASONS),
        force_refresh=force_refresh,
    )


def load_schedule_context():
    sched = load_schedules_cached()

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