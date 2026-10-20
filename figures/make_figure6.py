"""Figure 6: lesion size and performance in the two hospitals (exploratory, results/lesion_size.json).

(a) Distribution of the areas of the annotated lesions; (b) lesion-level sensitivity of the final model by lesion
area; (c) image-level DSC of the final model against the total lesion area of each lesion-positive view.
Hospital A is teal with solid lines and circles, Hospital B orange with dashed lines and squares, so colour is never
the only cue. Usage: python make_figure6.py <project_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT = Path(sys.argv[1])
INK, MUTED, GRID = "#12333D", "#61747B", "#D8E3E6"
STYLE = {"A": {"color": "#168B8A", "ls": "-", "marker": "o", "name": "Hospital A"},
         "B": {"color": "#C16636", "ls": "--", "marker": "s", "name": "Hospital B"}}
METHOD = "ours_ens"
MIN_N = 30  # bins with fewer lesions are drawn with open markers
AREA_EDGES = 10 ** np.arange(1.5, 4.75, 0.5)  # view total lesion area bins for the binned means in (c)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "font.weight": "bold",
                     "axes.labelweight": "bold", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})


def tidy(ax, title):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")


def main() -> None:
    res = json.loads((PROJECT / "results" / "lesion_size.json").read_text())
    m = res["methods"][METHOD]
    fig = plt.figure(figsize=(7.0, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.55, wspace=0.32)

    # (a) lesion areas
    ax = fig.add_subplot(gs[0, :])
    bins = 10 ** np.arange(0, 4.6, 0.1)
    for h, st in STYLE.items():
        areas = np.array(res["hospitals"][h]["areas"])
        w = np.full(areas.size, 100.0 / areas.size)
        ax.hist(areas, bins=bins, weights=w, histtype="stepfilled", color=st["color"], alpha=0.18)
        ax.hist(areas, bins=bins, weights=w, histtype="step", color=st["color"], linestyle=st["ls"], linewidth=2,
                label=f"{st['name']}: {areas.size:,} lesions, median {np.median(areas):.0f} pixels")
        ax.axvline(np.median(areas), ymax=0.72 if h == "A" else 1.0,  # stop below the legend
                   color=st["color"], linestyle=st["ls"], linewidth=1.4)
    ax.set_xscale("log")
    ax.set_xlim(1, 10 ** 4.5)
    ax.set_xlabel("Lesion area (pixels)")
    ax.set_ylabel("Lesions (%)")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    tidy(ax, "(a) Size of the annotated lesions")

    # (b) lesion-level sensitivity by size
    ax = fig.add_subplot(gs[1, 0])
    labels = res["size_bins"]
    x = np.arange(len(labels))
    for h, st in STYLE.items():
        rows = m["lesion_sensitivity"][h]["by_size"]
        y = np.array([np.nan if r["sensitivity"] is None else r["sensitivity"] for r in rows])
        n = np.array([r["n"] for r in rows])
        ax.plot(x, y, color=st["color"], linestyle=st["ls"], linewidth=2, label=st["name"])
        for xi, yi, ni in zip(x, y, n):
            if not np.isnan(yi):
                ax.plot(xi, yi, marker=st["marker"], markersize=7, color=st["color"],
                        markerfacecolor=st["color"] if ni >= MIN_N else "white", markeredgewidth=1.6)
    ax.set_xticks(x, [lab.replace("-", "–").replace(">=", "≥") for lab in labels], fontsize=9.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Lesion area (pixels)")
    ax.set_ylabel("Lesion-level sensitivity")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    tidy(ax, "(b) Detection by lesion size")

    # (c) per-view DSC against total lesion area
    ax = fig.add_subplot(gs[1, 1])
    for h, st in STYLE.items():
        area = np.array(m["view_area"][h], dtype=float)
        dsc = np.array(m["view_dsc"][h], dtype=float)
        ax.scatter(area, dsc, s=9, color=st["color"], alpha=0.35, marker=st["marker"], linewidths=0)
        idx = np.digitize(area, AREA_EDGES)
        centers, means = [], []
        for b in range(1, len(AREA_EDGES)):
            sel = idx == b
            if sel.sum() >= 5:
                centers.append(np.sqrt(AREA_EDGES[b - 1] * AREA_EDGES[b]))
                means.append(dsc[sel].mean())
        ax.plot(centers, means, color=st["color"], linestyle=st["ls"], linewidth=2, marker=st["marker"],
                markersize=6, label=st["name"])
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Total lesion area of the view (pixels)")
    ax.set_ylabel("DSC of the view")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    tidy(ax, "(c) Overlap by lesion burden")

    out = PROJECT / "figures" / "Figure6_lesion_size.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
