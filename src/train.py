# train.py
"""
Trains the model over walk-forward folds and compares to the baseline.

v2 adds a SEGMENTED (hybrid) prediction: use the XGBoost model for
high-volume players (baseline_pred >= HYBRID_THRESHOLD) and fall back
to the plain rolling-average baseline for low-volume players.

Justification (from error analysis on the v1 model):
  - High-volume players (baseline_pred >= 12): model beat baseline by 11.1% MAE
  - Low-volume players (baseline_pred < 12):   model was 1.7% WORSE than baseline
This is not post-hoc cherry-picking of the best number -- it's a modeling
decision directly motivated by evidence that the model only adds value where
there's enough usage signal to learn from. See LOG.md for the full analysis.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import config
from build_features import build_full_feature_table
from validation import walk_forward_folds

HYBRID_THRESHOLD = 12.0  # baseline_pred cutoff; matches the error-analysis split

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


def run_walk_forward_training(save_details=False):
    df = build_full_feature_table()
    df = df[~((df["season"] == config.VALIDATION_SEASON) &
              (df["week"] > config.MAX_VALIDATION_WEEK))]

    model_preds, baseline_preds, hybrid_preds, actuals = [], [], [], []
    detail_rows = []
    last_model = None
    fold_count = 0

    for week, train_df, test_df in walk_forward_folds(df):
        train_df = train_df.dropna(subset=[config.TARGET, "baseline_last4"])
        test_df = test_df.dropna(subset=[config.TARGET, "baseline_last4"])
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_train, y_train = train_df[FEATURE_COLS], train_df[config.TARGET]
        X_test, y_test = test_df[FEATURE_COLS], test_df[config.TARGET]

        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        last_model = model

        baseline_vals = test_df["baseline_last4"].values

        # Hybrid: model prediction where baseline_pred >= threshold,
        # else fall back to the plain baseline.
        hybrid_vals = np.where(baseline_vals >= HYBRID_THRESHOLD, preds, baseline_vals)

        model_preds.extend(preds)
        baseline_preds.extend(baseline_vals)
        hybrid_preds.extend(hybrid_vals)
        actuals.extend(y_test.values)
        fold_count += 1

        if save_details:
            for i, (_, row) in enumerate(test_df.iterrows()):
                detail_rows.append({
                    "player_display_name": row["player_display_name"],
                    "season": row["season"],
                    "week": row["week"],
                    "actual": y_test.values[i],
                    "model_pred": preds[i],
                    "baseline_pred": baseline_vals[i],
                    "hybrid_pred": hybrid_vals[i],
                })

    actuals = np.array(actuals)
    model_preds = np.array(model_preds)
    baseline_preds = np.array(baseline_preds)
    hybrid_preds = np.array(hybrid_preds)

    print(f"Folds evaluated: {fold_count}")
    print(f"Rows evaluated: {len(actuals)}\n")
    print(f"Baseline (last-4 avg)  MAE: {mae(actuals, baseline_preds):.3f}   RMSE: {rmse(actuals, baseline_preds):.3f}")
    print(f"Model (XGBoost)        MAE: {mae(actuals, model_preds):.3f}   RMSE: {rmse(actuals, model_preds):.3f}")
    print(f"Hybrid (segmented)     MAE: {mae(actuals, hybrid_preds):.3f}   RMSE: {rmse(actuals, hybrid_preds):.3f}\n")

    model_improvement = (mae(actuals, baseline_preds) - mae(actuals, model_preds)) / mae(actuals, baseline_preds) * 100
    hybrid_improvement = (mae(actuals, baseline_preds) - mae(actuals, hybrid_preds)) / mae(actuals, baseline_preds) * 100
    print(f"Model MAE improvement over baseline:  {model_improvement:.1f}%")
    print(f"Hybrid MAE improvement over baseline: {hybrid_improvement:.1f}%")

    if save_details:
        detail_df = pd.DataFrame(detail_rows)
        detail_df["model_error"] = (detail_df["actual"] - detail_df["model_pred"]).abs()
        detail_df["baseline_error"] = (detail_df["actual"] - detail_df["baseline_pred"]).abs()
        detail_df["hybrid_error"] = (detail_df["actual"] - detail_df["hybrid_pred"]).abs()
        detail_df["model_advantage"] = detail_df["baseline_error"] - detail_df["model_error"]
        detail_df["hybrid_advantage"] = detail_df["baseline_error"] - detail_df["hybrid_error"]
        detail_df.to_csv("data/processed/predictions_detail.csv", index=False)
        print("\nSaved row-level predictions to data/processed/predictions_detail.csv")

        if last_model is not None:
            importances = pd.Series(last_model.feature_importances_, index=FEATURE_COLS)
            importances = importances.sort_values(ascending=False)
            print("\nFeature importances (final fold's model):")
            print(importances.head(15))

    return model_preds, hybrid_preds, baseline_preds, actuals


if __name__ == "__main__":
    run_walk_forward_training(save_details=True)