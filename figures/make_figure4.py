"""Figure 4: qualitative examples (anterior view). Rows = patients, columns = ground truth and methods.
Overlay colors: true positive teal, false positive orange, false negative deep blue.
Usage: python make_figure4.py <project_dir> <cases_npz>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.ndimage import binary_erosion  # noqa: E402

PROJECT, CASES = Path(sys.argv[1]), Path(sys.argv[2])
INK = "#12333D"
TP, FP, FN, GT = to_rgb("#168B8A"), to_rgb("#C16636"), to_rgb("#12333D"), to_rgb("#168B8A")
COLS = [("gt", "Ground truth"), ("ours_ens", "Ours"), ("ours_anat", "Single network,\nanatomy task"),
        ("bsciseg", "BSci-Seg"), ("nnunet", "nnU-Net")]
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12, "font.weight": "bold"})

z = np.load(CASES)
n_cases = len([k for k in z.files if k.endswith("_image")])
cols = [c for c in COLS if c[0] == "gt" or f"c0_{c[0]}" in z.files]
fig, axes = plt.subplots(n_cases, len(cols), figsize=(2.3 * len(cols), 6.2 * n_cases))
axes = np.atleast_2d(axes)
for i in range(n_cases):
    img = z[f"c{i}_image"]
    gt = z[f"c{i}_gt"].astype(bool)
    meta = json.loads(str(z[f"c{i}_meta"]))
    dices = json.loads(str(z[f"c{i}_dice"]))
    hi = np.percentile(img[img > 0], 99) if np.any(img > 0) else 1.0
    base = 1.0 - np.clip(img / hi, 0, 1)
    rows = np.where(base.min(axis=1) < 0.97)[0]
    top, bottom = (max(rows.min() - 10, 0), min(rows.max() + 10, img.shape[0])) if len(rows) else (0, img.shape[0])
    for j, (key, title) in enumerate(cols):
        ax = axes[i, j]
        rgb = np.repeat(base[..., None], 3, axis=2)
        if key == "gt":
            rgb[gt] = 0.35 * rgb[gt] + 0.65 * np.array(GT)
            label = {"A": "CMUH", "B": "TTMHH"}[meta["hospital"]]  # hospital keys used in cases.npz
        else:
            pred = z[f"c{i}_{key}"].astype(bool)
            rgb[pred] = 0.35 * rgb[pred] + 0.65 * np.array(FP)
            edge = gt & ~binary_erosion(gt, iterations=2)
            rgb[edge] = np.array(TP)
            d = dices.get(key)
            label = f"Dice {d:.2f}" if d is not None else ""
        ax.imshow(rgb[top:bottom], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if i == 0:
            ax.set_title(title, color=INK, fontsize=12, fontweight="bold")
        ax.text(0.5, -0.02, label, transform=ax.transAxes, ha="center", va="top", color=INK, fontsize=11)
handles = [Patch(color=TP, label="Reference lesion (filled in first column, outlined elsewhere)"),
           Patch(color=FP, label="Model prediction")]
fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, fontsize=12)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = PROJECT / "figures" / "Figure4_examples.png"
fig.savefig(out, dpi=300)
print("saved", out)
