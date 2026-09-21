# validation.py
"""
Walk-forward validation: for each week in the validation season,
train on everything strictly before it, predict that week.

This is the core safeguard against leakage (see spec Part A3).
NEVER replace this with a random/shuffled split.
"""
import config

def walk_forward_folds(df, validation_season=None):
    """
    Yields (train_df, test_df) pairs.
    train_df = every row strictly before the target week
    test_df  = exactly the rows for that target week
    """
    validation_season = validation_season or config.VALIDATION_SEASON

    weeks = sorted(
        df.loc[df["season"] == validation_season, "week"].unique()
    )

    for week in weeks:
        # Train: everything from train seasons, PLUS validation-season
        # weeks strictly before this one (so the walk actually advances
        # within the season, not just season-to-season).
        train_mask = (
            (df["season"] < validation_season) |
            ((df["season"] == validation_season) & (df["week"] < week))
        )
        test_mask = (df["season"] == validation_season) & (df["week"] == week)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(test_df) == 0 or len(train_df) == 0:
            continue

        yield week, train_df, test_df


