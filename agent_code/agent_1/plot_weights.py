import json
import os
import matplotlib.pyplot as plt
 

WEIGHTS_DIR = "." 
 
# Map each file name to the cumulative round number it represents
FILE_TO_ROUND = {
    "weights_0.json": 620,
    "weights_1.json": 920,
}
 
for r in range(1000, 5800 + 1, 200):
    FILE_TO_ROUND[f"weights_round_{r}.json"] = r
 
PHASES = [
    (0,   20,  "Phase 1: coin-heaven"),
    (20,  320, "Phase 2: classic"),
    (320, 620, "Phase 3: classic vs 3 rule_based"),
]
 
# Learning-rate schedule: (start_round, end_round, lr)
LR_SCHEDULE = [
    (0,    920,  0.01),
    (920,  3920, 0.03),
    (3920, 5800, 0.003),
]
LR_COLORS = {0.01: "#dfefff", 0.03: "#fff3d6", 0.003: "#e6f7e0"}
 

data = {}  # round -> {feature: value}
for fname, round_num in FILE_TO_ROUND.items():
    path = os.path.join(WEIGHTS_DIR, fname)
    if not os.path.exists(path):
        print(f"Warning: {path} not found, skipping.")
        continue
    with open(path, "r") as f:
        data[round_num] = json.load(f)
 
if not data:
    raise SystemExit("No weight files were found. Check WEIGHTS_DIR.")
 
rounds = sorted(data.keys())
feature_names = list(next(iter(data.values())).keys())
 
# round -> value for each feature, in round order
series = {feat: [data[r][feat] for r in rounds] for feat in feature_names}
 

fig, ax = plt.subplots(figsize=(14, 8))
 
# Shade the learning-rate regions
max_round = max(rounds)
for start, end, lr in LR_SCHEDULE:
    end_clipped = min(end, max_round)
    if start >= max_round:
        continue
    ax.axvspan(start, end_clipped, color=LR_COLORS.get(lr, "#eeeeee"), alpha=0.5, zorder=0)
 
# Add one legend entry per distinct learning rate (avoid duplicate labels)
seen_lr = set()
for start, end, lr in LR_SCHEDULE:
    if lr not in seen_lr:
        ax.axvspan(0, 0, color=LR_COLORS.get(lr, "#eeeeee"), alpha=0.5, label=f"lr = {lr}")
        seen_lr.add(lr)
 
# Plot each feature's weight trajectory in a distinct color
cmap = plt.get_cmap("tab10")
for i, feat in enumerate(feature_names):
    ax.plot(rounds, series[feat], marker="o", markersize=4,
            color=cmap(i % 10), label=feat, linewidth=1.8, zorder=3)
 
# Mark phase boundaries (the three initial training commands)
for start, end, label in PHASES:
    ax.axvline(end, color="gray", linestyle="--", linewidth=1, zorder=2)
    #ax.text(end, ax.get_ylim()[1], label, rotation=90, va="top", ha="left", fontsize=7, color="dimgray")
    ax.annotate(label, xy=(end, 1), xycoords=("data", "axes fraction"),
                xytext=(5, -8), textcoords="offset points",
                rotation=90, va="top", ha="left",
                fontsize=11, color="dimgray")

 
ax.set_xlabel("Training rounds", fontsize=13)
ax.set_ylabel("Weight value", fontsize=13)
ax.set_xlim(0, max(rounds)+10)
ax.axhline(0, color="black", linewidth=0.6)
ax.tick_params(axis="both", labelsize=11)
ax.legend(loc="upper left", bbox_to_anchor=(0.468, 0.9),fontsize=12, framealpha=0.9, ncol=4, columnspacing=1.0, handletextpad=0.5)

ax.grid(True, alpha=0.3)
 
fig.tight_layout()
fig.savefig("weights_over_time.png", dpi=200, bbox_inches="tight")
