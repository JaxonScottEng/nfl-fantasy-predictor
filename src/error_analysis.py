# error_analysis.py
import pandas as pd

detail = pd.read_csv("data/processed/predictions_detail.csv")

# Use baseline_pred as a proxy for "how established/high-volume is this player,"
# since it's just their own recent scoring level -- a clean, already-available
# grouping variable, no extra computation needed.
THRESHOLD = 12.0

def summarize(name, group):
    model_mae = group["model_error"].mean()
    baseline_mae = group["baseline_error"].mean()
    improvement = (baseline_mae - model_mae) / baseline_mae * 100
    print(f"{name}: n={len(group)}")
    print(f"  Baseline MAE: {baseline_mae:.3f}")
    print(f"  Model MAE:    {model_mae:.3f}")
    print(f"  Improvement:  {improvement:.1f}%\n")

# Split per position -- positions have different scoring distributions, so
# pooling them would average away the segment effect this analysis exists to find.
for position in sorted(detail["position"].unique()):
    pos_detail = detail[detail["position"] == position]
    print(f"########## {position} ##########")
    summarize("HIGH-VOLUME (baseline_pred >= 12)", pos_detail[pos_detail["baseline_pred"] >= THRESHOLD])
    summarize("LOW-VOLUME  (baseline_pred < 12)", pos_detail[pos_detail["baseline_pred"] < THRESHOLD])