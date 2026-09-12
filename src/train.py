# train.py
"""
Trains the first real model (gradient-boosted trees) over the walk-forward
folds and compares its error to the Phase 1 baseline, on the SAME rows,
in the SAME folds. No shuffled split, no leakage -- see validation.py.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import config
from build_features import build_full_feature_table
from validation import walk_forward_folds

# Feature columns: everything rolling/lagged, plus the baseline itself as
# a feature (per spec: "rolling fantasy points" is a legitimate, informative
# feature, not cheating, since it's already lagged).
FEATURE_COLS = [
    "targets_roll3", "targets_roll4", "targets_roll5",
    "receptions_roll3", "receptions_roll4", "receptions_roll5",
    "receiving_air_yards_roll3", "receiving_air_yards_roll4", "receiving_air_yards_roll5",
    "carries_roll3", "carries_roll4", "carries_roll5",
    "receiving_yards_roll3", "receiving_yards_roll4", "receiving_yards_roll5",
    "rushing_yards_roll3", "rushing_yards_roll4", "rushing_yards_roll5",
    "receiving_tds_roll3", "receiving_tds_roll4", "receiving_tds_roll5",
    "rushing_tds_roll3", "rushing_tds_roll4", "rushing_tds_roll5",
    "yards_per_target_roll3", "yards_per_target_roll4", "yards_per_target_roll5",
    "def_points_allowed_roll3", "def_points_allowed_roll4", "def_points_allowed_roll5",
    "implied_total", "spread_line", "total_line",
    "baseline_last4",
]

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def run_walk_forward_training():
    df = build_full_feature_table()

    # Playoff weeks excluded from validation (see config + Phase 2 note).
    df = df[~((df["season"] == config.VALIDATION_SEASON) &
              (df["week"] > config.MAX_VALIDATION_WEEK))]

    model_preds, baseline_preds, actuals = [], [], []
    fold_count = 0

    for week, train_df, test_df in walk_forward_folds(df):
        # Drop rows with no target or no baseline yet (first game of a career).
        train_df = train_df.dropna(subset=[config.TARGET, "baseline_last4"])
        test_df = test_df.dropna(subset=[config.TARGET, "baseline_last4"])
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_train, y_train = train_df[FEATURE_COLS], train_df[config.TARGET]
        X_test, y_test = test_df[FEATURE_COLS], test_df[config.TARGET]

        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        model_preds.extend(preds)
        baseline_preds.extend(test_df["baseline_last4"].values)
        actuals.extend(y_test.values)
        fold_count += 1

    actuals = np.array(actuals)
    model_preds = np.array(model_preds)
    baseline_preds = np.array(baseline_preds)

    print(f"Folds evaluated: {fold_count}")
    print(f"Rows evaluated: {len(actuals)}")
    print()
    print(f"Baseline (last-4 avg)  MAE: {mae(actuals, baseline_preds):.3f}   RMSE: {rmse(actuals, baseline_preds):.3f}")
    print(f"Model (XGBoost)        MAE: {mae(actuals, model_preds):.3f}   RMSE: {rmse(actuals, model_preds):.3f}")
    print()
    improvement = (mae(actuals, baseline_preds) - mae(actuals, model_preds)) / mae(actuals, baseline_preds) * 100
    print(f"MAE improvement over baseline: {improvement:.1f}%")

if __name__ == "__main__":
    run_walk_forward_training()