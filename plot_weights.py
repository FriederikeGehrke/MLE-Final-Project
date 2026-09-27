# Plot Agent 2's average feature sensitivity

import glob
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt

from agent_code.agent_1.callbacks import FEATURES, action_features
from environment import BombeRLeWorld, WorldArgs

WEIGHTS_DIR = "agent_code/agent_2"

# Map each .npz file to the cumulative training round it represents
FILE_TO_ROUND = {
    "weights_1.npz": 20,
    "weights_2.npz": 320,
    "weights_4.npz": 2420,
    "weights_5.npz": 5420,
    "weights_6.npz": 5920,
}


PHASES = [
    (0, 20, "Phase 1: coin-heaven"),
    (20, 320, "Phase 2: classic"),
    #(320, 620, "Phase 3: classic vs 3 rule_based"),
]


def collect_states(n_rounds=15, scenario="classic"):

    args = WorldArgs(
        no_gui=True,
        fps=15,
        turn_based=False,
        update_interval=0.1,
        save_replay=False,
        replay=None,
        make_video=False,
        continue_without_training=True,
        log_dir="logs",
        save_stats=False,
        match_name=None,
        seed=1,
        silence_errors=False,
        scenario=scenario,
    )

    agents = [
        ("agent_1", False),
        ("rule_based_agent", False),
        ("rule_based_agent", False),
        ("rule_based_agent", False),
    ]

    world = BombeRLeWorld(args, agents)
    states = []

    for _ in range(n_rounds):
        world.new_round()
        while world.running:
            world.do_step()
            gs = world.get_state_for_agent(world.agents[0])
            if gs is not None:
                states.append(gs)
    world.end()

    return states


def mean_gradient(params, feats_list):

    grads = []

    for feats, valid in feats_list:
        for i in valid:
            phi = feats[i]
            z1 = phi @ params["W1"].T + params["b1"]
            h = np.tanh(z1)
            grads.append((params["W2"] * (1 - h ** 2)) @ params["W1"])

    return np.mean(grads, axis=0)


states = collect_states(n_rounds=15)
feats_list = []

for gs in states:
    feats, valid, doomed = action_features(gs)
    feats_list.append((feats, np.flatnonzero(valid)))


# Special checkpoint files and the rounds they represent.
SPECIAL_CHECKPOINTS = {
    "weights_1.npz": 20,
    "weights_2.npz": 320,
    "weights_4.npz": 2420,
    "weights_5.npz": 5420,
    "weights_6.npz": 5920,
}

CHECKPOINT_DIR = "agent_code/agent_2"

rounds = []
sensitivities = []

for filename, round_num in SPECIAL_CHECKPOINTS.items():

    path = f"{CHECKPOINT_DIR}/{filename}"
    if not os.path.exists(path):
        print(f"Warning: {path} not found, skipping.")
        continue

    print(f"Loading {path} -> round {round_num}")

    mlp = np.load(path)
    params = {k: mlp[k] for k in mlp.files}
    rounds.append(round_num)
    sensitivities.append(mean_gradient(params, feats_list))


round_checkpoint_files = glob.glob(f"{CHECKPOINT_DIR}/weights_round_*.npz")

for path in round_checkpoint_files:

    match = re.search(r"weights_round_(\d+)\.npz$", path)
    if match is None:
        continue

    round_num = int(match.group(1))
    mlp = np.load(path)

    params = {k: mlp[k] for k in mlp.files}
    rounds.append(round_num)

    sensitivities.append(mean_gradient(params, feats_list))


if not sensitivities:
    raise SystemExit("No checkpoint files were found.")

order = np.argsort(rounds)
rounds = np.asarray(rounds)[order]
sensitivities = np.asarray(sensitivities)[order]


with open("agent_code/agent_1/weights.json") as f:
    lin_w = json.load(f)


# Create plot
fig, ax = plt.subplots(figsize=(14, 8))

cmap = plt.get_cmap("tab10")

for i, feature_name in enumerate(FEATURES):
    ax.plot(
        rounds,
        sensitivities[:, i],
        marker="o",
        markersize=4,
        color=cmap(i % 10),
        label=feature_name,
        linewidth=1.8,
        zorder=3,
    )


# for i, feature_name in enumerate(FEATURES):

#     if feature_name in lin_w:
#         ax.axhline(
#             lin_w[feature_name],
#             color=cmap(i % 10),
#             linestyle="--",
#             linewidth=1,
#             alpha=0.35,
#             zorder=1,
#         )


max_round = max(rounds)

for start, end, label in PHASES:
    if end > max_round:
        continue
    
    ax.axvline(
        end,
        color="gray",
        linestyle="--",
        linewidth=1,
        zorder=2,
    )

    ax.annotate(
        label,
        xy=(end, 1),
        xycoords=("data", "axes fraction"),
        xytext=(5, -8),
        textcoords="offset points",
        rotation=90,
        va="top",
        ha="left",
        fontsize=11,
        color="dimgray",
    )

ax.set_xlabel("Training rounds", fontsize=13)
ax.set_ylabel("Mean feature sensitivity", fontsize=13)
ax.set_xlim(0, max(rounds) + 10)
ax.axhline(0, color="black", linewidth=0.6)
ax.tick_params(axis="both", labelsize=11)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.58, 0.87),
    fontsize=12,
    framealpha=0.9,
    ncol=3,
    columnspacing=1.0,
    handletextpad=0.5,
)

ax.grid(True,alpha=0.3,)

fig.tight_layout()

fig.savefig("sensitivity_evolution.png", dpi=200, bbox_inches="tight")

