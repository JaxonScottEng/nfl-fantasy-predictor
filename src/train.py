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

v3 trains one SEPARATE model per position in config.ACTIVE_POSITIONS.
Positions are never pooled into a single model: usage stats mean different
things by position (a WR's carries vs an RB's), and pooling would let one
position's sample size and scoring distribution distort the other's fit.
Each position gets its own feature list, its own folds, and its own
baseline/model/hybrid comparison on its own rows.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import config
from build_features import build_full_feature_table
from metrics import mae, rmse
from validation import walk_forward_folds

# Single source of truth for the model spec, so evaluation comparators can never
# silently drift from what training actually fits.
XGB_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)

# baseline_pred cutoff; matches the error-analysis split.
# WR's 12.0 comes from the Phase 5 WR error analysis. RB inherits that same
# cutoff rather than a separately tuned one -- picking RB's threshold by
# scanning for its best result would be exactly the post-hoc cherry-picking
# the WR analysis avoided. Revisit only off a dedicated RB error analysis.
HYBRID_THRESHOLD_BY_POSITION = {
    "WR": 12.0,
    "RB": 12.0,
}
DEFAULT_HYBRID_THRESHOLD = 12.0

# Receiving-led. Unchanged from the locked-in WR model -- do not edit
# without re-verifying the numbers in CLAUDE.md's Verification section.
WR_FEATURE_COLS = [
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

# Rushing-led, plus the receiving side that PPR makes matter for backs.
# Adds touches (carries + receptions), yards-per-carry, and rushing first
# downs -- the RB analogues of the WR list's target/yards-per-target signals.
RB_FEATURE_COLS = [
    "carries_roll3", "carries_roll4", "carries_roll5",
    "touches_roll3", "touches_roll4", "touches_roll5",
    "targets_roll3", "targets_roll4", "targets_roll5",
    "receptions_roll3", "receptions_roll4", "receptions_roll5",
    "rushing_yards_roll3", "rushing_yards_roll4", "rushing_yards_roll5",
    "receiving_yards_roll3", "receiving_yards_roll4", "receiving_yards_roll5",
    "rushing_tds_roll3", "rushing_tds_roll4", "rushing_tds_roll5",
    "receiving_tds_roll3", "receiving_tds_roll4", "receiving_tds_roll5",
    "rushing_first_downs_roll3", "rushing_first_downs_roll4", "rushing_first_downs_roll5",
    "yards_per_carry_roll3", "yards_per_carry_roll4", "yards_per_carry_roll5",
    "yards_per_target_roll3", "yards_per_target_roll4", "yards_per_target_roll5",
    "def_points_allowed_roll3", "def_points_allowed_roll4", "def_points_allowed_roll5",
    "implied_total", "spread_line", "total_line",
    "baseline_last4",
]

FEATURE_COLS_BY_POSITION = {
    "WR": WR_FEATURE_COLS,
    "RB": RB_FEATURE_COLS,
}


def run_position(df, position, save_details=False):
    """Walk-forward train/evaluate a single position on its own rows."""
    feature_cols = FEATURE_COLS_BY_POSITION[position]
    threshold = HYBRID_THRESHOLD_BY_POSITION.get(position, DEFAULT_HYBRID_THRESHOLD)
    pos_df = df[df["position"] == position]

    model_preds, baseline_preds, hybrid_preds, actuals = [], [], [], []
    detail_rows = []
    last_model = None
    fold_count = 0

    for week, train_df, test_df in walk_forward_folds(pos_df):
        train_df = train_df.dropna(subset=[config.TARGET, "baseline_last4"])
        test_df = test_df.dropna(subset=[config.TARGET, "baseline_last4"])
        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_train, y_train = train_df[feature_cols], train_df[config.TARGET]
        X_test, y_test = test_df[feature_cols], test_df[config.TARGET]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        last_model = model

        baseline_vals = test_df["baseline_last4"].values

        # Hybrid: model prediction where baseline_pred >= threshold,
        # else fall back to the plain baseline.
        hybrid_vals = np.where(baseline_vals >= threshold, preds, baseline_vals)

        model_preds.extend(preds)
        baseline_preds.extend(baseline_vals)
        hybrid_preds.extend(hybrid_vals)
        actuals.extend(y_test.values)
        fold_count += 1

        if save_details:
            for i, (_, row) in enumerate(test_df.iterrows()):
                detail_rows.append({
                    "player_display_name": row["player_display_name"],
                    "position": position,
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

    print(f"===== {position} =====")
    print(f"Folds evaluated: {fold_count}")
    print(f"Rows evaluated: {len(actuals)}\n")
    print(f"{position} Baseline (last-4 avg)  MAE: {mae(actuals, baseline_preds):.3f}   RMSE: {rmse(actuals, baseline_preds):.3f}")
    print(f"{position} Model (XGBoost)        MAE: {mae(actuals, model_preds):.3f}   RMSE: {rmse(actuals, model_preds):.3f}")
    print(f"{position} Hybrid (segmented)     MAE: {mae(actuals, hybrid_preds):.3f}   RMSE: {rmse(actuals, hybrid_preds):.3f}\n")

    model_improvement = (mae(actuals, baseline_preds) - mae(actuals, model_preds)) / mae(actuals, baseline_preds) * 100
    hybrid_improvement = (mae(actuals, baseline_preds) - mae(actuals, hybrid_preds)) / mae(actuals, baseline_preds) * 100
    print(f"{position} Model MAE improvement over baseline:  {model_improvement:.1f}%")
    print(f"{position} Hybrid MAE improvement over baseline: {hybrid_improvement:.1f}%\n")

    if save_details and last_model is not None:
        importances = pd.Series(last_model.feature_importances_, index=feature_cols)
        importances = importances.sort_values(ascending=False)
        print(f"{position} feature importances (final fold's model):")
        print(importances.head(15))
        print()

    return {
        "position": position,
        "model_preds": model_preds,
        "hybrid_preds": hybrid_preds,
        "baseline_preds": baseline_preds,
        "actuals": actuals,
        "detail_rows": detail_rows,
    }


def run_walk_forward_training(save_details=False):
    df = build_full_feature_table()
    df = df[~((df["season"] == config.VALIDATION_SEASON) &
              (df["week"] > config.MAX_VALIDATION_WEEK))]

    results = {}
    all_detail_rows = []
    for position in config.ACTIVE_POSITIONS:
        result = run_position(df, position, save_details=save_details)
        results[position] = result
        all_detail_rows.extend(result["detail_rows"])

    if save_details and all_detail_rows:
        detail_df = pd.DataFrame(all_detail_rows)
        detail_df["model_error"] = (detail_df["actual"] - detail_df["model_pred"]).abs()
        detail_df["baseline_error"] = (detail_df["actual"] - detail_df["baseline_pred"]).abs()
        detail_df["hybrid_error"] = (detail_df["actual"] - detail_df["hybrid_pred"]).abs()
        detail_df["model_advantage"] = detail_df["baseline_error"] - detail_df["model_error"]
        detail_df["hybrid_advantage"] = detail_df["baseline_error"] - detail_df["hybrid_error"]
        detail_df.to_csv("data/processed/predictions_detail.csv", index=False)
        print("Saved row-level predictions to data/processed/predictions_detail.csv")

    return results


if __name__ == "__main__":
    run_walk_forward_training(save_details=True)
