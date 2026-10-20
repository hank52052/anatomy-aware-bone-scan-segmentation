"""Supplementary Figure S1: training curves (results/histories/*.json).

(a) Validation lesion Dice per epoch (the value used to keep the best checkpoint) for every fold; thick line =
mean over the folds still training. (b) Training loss of our networks. One epoch is five passes over the
training patients for our networks and ten for BSci-Seg.
Usage: python make_figureS1.py <project_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT = Path(sys.argv[1])
HIST = PROJECT / "results" / "histories"
INK, MUTED, GRID, OURS = "#12333D", "#61747B", "#D8E3E6", "#C16636"
RUNS = [("final_ours_anat", "Single network, anatomy task (MAE)", OURS, "-"),
        ("final_ours_noanat", "Single network, no anatomy task (MAE)", OURS, ":"),
        ("final_ours_anat_nomae", "Single network, anatomy task, no MAE", INK, "--"),
        ("ablation_BSciSeg", "BSci-Seg (re-implemented)", MUTED, "-.")]
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12, "font.weight": "bold",
                     "axes.labelweight": "bold", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})


def load(run, fold):
    p = HIST / f"{run}_fold{fold}.json"
    if not p.exists():
        return None
    h = json.loads(p.read_text())
    if isinstance(h, list):  # BSci-Seg: list of per-epoch dicts
        h = {k: [e.get(k) for e in h] for k in h[0]}
    return h


def mean_curve(curves):
    n = max(len(c) for c in curves)
    return np.array([np.mean([c[i] for c in curves if i < len(c)]) for i in range(n)])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
for run, label, color, style in RUNS:
    hs = [h for h in (load(run, f) for f in range(5)) if h]
    if not hs:
        continue
    for key, ax in (("val_lesion_dice", ax1), ("train_loss", ax2)):
        if key == "train_loss" and run == "ablation_BSciSeg":
            continue  # different loss terms, not comparable
        curves = [np.array(h[key], dtype=float) for h in hs]
        for c in curves:
            ax.plot(np.arange(1, len(c) + 1), c, color=color, linestyle=style, linewidth=0.8, alpha=0.35)
        m = mean_curve(curves)
        ax.plot(np.arange(1, len(m) + 1), m, color=color, linestyle=style, linewidth=2.2,
                label=f"{label} ({len(hs)} folds)")
ax1.set_ylabel("Validation lesion Dice"); ax2.set_ylabel("Training loss")
ax1.set_title("(a) Validation fold", loc="left", color=INK, fontweight="bold")
ax2.set_title("(b) Training loss of our networks", loc="left", color=INK, fontweight="bold")
for ax in (ax1, ax2):
    ax.set_xlabel("Epoch")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
out = PROJECT / "figures" / "FigureS1_training_curves.png"
fig.savefig(out, dpi=300)
print("saved", out)
