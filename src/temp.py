import pandas as pd
detail = pd.read_csv("data/processed/predictions_detail.csv")

print("=== Model's best wins over baseline ===")
print(detail.sort_values("model_advantage", ascending=False).head(10)[
    ["player_display_name", "week", "actual", "model_pred", "baseline_pred", "model_advantage"]
])

print("\n=== Model's worst losses to baseline ===")
print(detail.sort_values("model_advantage", ascending=True).head(10)[
    ["player_display_name", "week", "actual", "model_pred", "baseline_pred", "model_advantage"]
])

print("\n=== Per-player average advantage (best) ===")
per_player = detail.groupby("player_display_name")["model_advantage"].mean().sort_values(ascending=False)
print(per_player.head(10))

print("\n=== Per-player average advantage (worst) ===")
print(per_player.tail(10))