"""Figure 5: patient-level ROC curves for detecting bone metastases (results/patient_auc.json).

Score = sum of the lesion probability over both views. Colour separates our models (orange), nnU-Net (ink) and
BSci-Seg (gray); line style and the legend (with AUC) carry identity, so colour is never the only cue.
Usage: python make_figure5.py <project_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT = Path(sys.argv[1])
INK, MUTED, GRID, OURS = "#12333D", "#61747B", "#D8E3E6", "#C16636"
CURVES = [("ours_ens", "Ours", OURS, "-", 2.6), ("ours_anat", "Single network, anatomy task", OURS, "--", 1.6),
          ("ours_noanat", "Single network, no anatomy task", OURS, ":", 1.8),
          ("bsciseg", "BSci-Seg (re-implemented)", MUTED, "-.", 1.8), ("nnunet", "nnU-Net", INK, "-", 1.8)]
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12, "font.weight": "bold",
                     "axes.labelweight": "bold", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})

res = json.loads((PROJECT / "results" / "patient_auc.json").read_text())
fig, ax = plt.subplots(figsize=(6.4, 6.0))
ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.2, zorder=0)
for key, label, color, style, width in CURVES:
    m = res["methods"][key]
    ax.plot(m["roc"]["fpr"], m["roc"]["tpr"], color=color, linestyle=style, linewidth=width,
            label=f"{label}: {m['auc']:.3f} ({m['ci'][0]:.3f} to {m['ci'][1]:.3f})")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
ax.set_xlabel("1 − specificity")
ax.set_ylabel("Sensitivity")
ax.set_title(f"Patients with ({res['n_positive']}) and without ({res['n_negative']}) bone metastases",
             loc="left", color=INK, fontsize=12, fontweight="bold")
ax.grid(color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(title="AUC (95% CI)", loc="lower right", frameon=False, fontsize=10, title_fontsize=10)
fig.tight_layout()
out = PROJECT / "figures" / "Figure5_roc.png"
fig.savefig(out, dpi=300)
print("saved", out)
