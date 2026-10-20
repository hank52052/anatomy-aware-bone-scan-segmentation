"""Figure 3: (a) per-patient Dice by method, (b) lesion-free scans with any false positive.

Reads results/per_patient_dsc.csv and results/lesion_free_fp.csv. Identity is carried by the
x-axis labels; colour only marks our model (orange) against the comparison methods (gray).
Usage: python make_figure3.py <project_dir>
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT = Path(sys.argv[1])
RES = PROJECT / "results"
INK, MUTED, GRID, OURS, OTHER = "#12333D", "#61747B", "#D8E3E6", "#C16636", "#61747B"
ORDER = [("ours_ens", "Ours"), ("ours_anat", "Single network,\nanatomy task"),
         ("ours_noanat", "Single network,\nno anatomy task"), ("bsciseg", "BSci-Seg\n(re-implemented)"),
         ("nnunet", "nnU-Net")]
FINAL = "ours_ens"  # our final model: mean of the networks with and without the anatomy task

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12, "font.weight": "bold",
                     "axes.labelweight": "bold", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})

with open(RES / "per_patient_dsc.csv", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
dsc = {k: np.array([float(r[k]) for r in rows if r.get(k) not in (None, "", "nan")]) for k, _ in ORDER
       if rows and k in rows[0]}
with open(RES / "lesion_free_fp.csv", encoding="utf-8") as fh:
    lf_rows = list(csv.DictReader(fh))
lf = defaultdict(list)
for r in lf_rows:
    lf[r["method"]].append(int(r["fp_components"]))

methods = [(k, lab) for k, lab in ORDER if k in dsc]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
rng = np.random.default_rng(0)
for i, (k, _) in enumerate(methods):
    color = OURS if k == FINAL else OTHER
    v = dsc[k]
    ax1.boxplot(v, positions=[i], widths=0.5, showfliers=False, medianprops={"color": color, "linewidth": 2},
                boxprops={"color": color, "linewidth": 1.5}, whiskerprops={"color": color, "linewidth": 1.2},
                capprops={"color": color, "linewidth": 1.2})
    ax1.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=9, color=color, alpha=0.35, linewidths=0, zorder=1)
    ax1.scatter([i], [v.mean()], marker="D", s=40, color="white", edgecolors=color, linewidths=1.6, zorder=3)
    ax1.text(i, 1.03, f"{v.mean():.3f}", ha="center", va="bottom", color=INK, fontsize=11)
ax1.set_xticks(range(len(methods)), [lab for _, lab in methods], fontsize=10)
ax1.set_ylim(0, 1.1)
ax1.set_yticks(np.arange(0, 1.01, 0.2))
ax1.set_ylabel("Per-patient Dice")
ax1.set_title("(a) Lesion-positive patients", loc="left", color=INK, fontweight="bold")
ax1.grid(axis="y", color=GRID, linewidth=0.8)
ax1.set_axisbelow(True)

lf_methods = [(k, lab) for k, lab in ORDER if lf.get(k)]
for i, (k, _) in enumerate(lf_methods):
    counts = np.array(lf[k])
    pct = (counts > 0).mean() * 100
    color = OURS if k == FINAL else OTHER
    ax2.bar(i, pct, width=0.55, color=color, edgecolor="white", linewidth=2)
    ax2.text(i, pct + 2, f"{pct:.0f}%\n(mean {counts.mean():.1f})", ha="center", va="bottom", color=INK, fontsize=10)
ax2.set_xticks(range(len(lf_methods)), [lab for _, lab in lf_methods], fontsize=10)
ax2.set_ylim(0, 118)
ax2.set_ylabel("Lesion-free scans with ≥1 false positive (%)")
ax2.set_title(f"(b) Lesion-free patients (n = {len(lf[lf_methods[0][0]]) if lf_methods else 0})", loc="left",
              color=INK, fontweight="bold")
ax2.grid(axis="y", color=GRID, linewidth=0.8)
ax2.set_axisbelow(True)
for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout()
out = PROJECT / "figures" / "Figure3_results.png"
fig.savefig(out, dpi=300)
print("saved", out)
