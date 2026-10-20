"""High-tier AI-journal-style plotting helpers (Nature / NeurIPS / CVPR).

Default DPI = 1200 (raster).  Uses seaborn 'ticks' theme + 'paper' context for
publication-grade typography; matplotlib axes are de-spined; consistent
qualitative palette across the whole pipeline.

All public functions save figures at 1200 DPI by default. Caller can override
via `dpi=`. Vector format also produced when path ends in .pdf/.svg.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DEFAULT_DPI = 1200
FONT_STACK = ["Arial", "Helvetica", "Microsoft JhengHei", "Microsoft YaHei",
              "SimHei", "DejaVu Sans"]
PALETTE = sns.color_palette("Set2", 8)            # color-blind safe, journal style
SEQ_CMAP = sns.color_palette("rocket_r", as_cmap=True)
DIVERG_CMAP = sns.color_palette("vlag", as_cmap=True)


def _rc():
    """High-tier journal style — seaborn 'ticks' + paper context + clean axes."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.15,
                  rc={"axes.grid": False})
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = FONT_STACK
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["lines.linewidth"] = 1.4
    plt.rcParams["patch.linewidth"] = 0.6
    plt.rcParams["pdf.fonttype"] = 42              # embed TrueType (editable in Illustrator)
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.transparent"] = False
    plt.rcParams["figure.dpi"] = 150               # display; save overrides to DEFAULT_DPI


def _save(fig, path: Path, dpi: int = DEFAULT_DPI, also_vector: bool = True) -> None:
    """Save raster at `dpi`; if also_vector, save matching .pdf alongside."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if also_vector and path.suffix.lower() == ".png":
        try:
            fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        except Exception:
            pass
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loss / metric curves
# ---------------------------------------------------------------------------
def plot_training_curves(history: dict, out_path: Path, *,
                         dpi: int = DEFAULT_DPI,
                         **_legacy) -> None:
    """4-panel diagnostic of training run.

    history keys we read (all optional, plotted only if present and non-empty):
      epoch                  — x axis
      train_loss             — top-left: total training loss
      val_lesion_dice        — top-right: validation lesion Dice (★ headline)
      val_anatomy_dice       — top-right secondary (if any non-nan)
      train_les_dice_loss    — bottom-right: per-term loss breakdown
      train_les_focal        — bottom-right
      train_ana_dice_loss    — bottom-right
      train_ana_cons         — bottom-right
      lr                     — bottom-left primary
      alpha                  — bottom-left secondary (HBS gate curriculum)
    """
    import numpy as _np
    _rc()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    ax_loss, ax_dice = axes[0]
    ax_lr,   ax_terms = axes[1]
    epoch = history.get("epoch", list(range(len(history.get("train_loss", [])))))

    # ---- top-left: train_loss ----
    train_loss = history.get("train_loss", [])
    if train_loss:
        ax_loss.plot(epoch[:len(train_loss)], train_loss, color=PALETTE[0],
                     linewidth=1.6, label="train total loss")
        ax_loss.scatter([epoch[-1]], [train_loss[-1]], color=PALETTE[0], s=18, zorder=5)
        ax_loss.text(epoch[-1], train_loss[-1],
                     f"  {train_loss[-1]:.3f}", fontsize=8, va="center")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss", fontweight="bold")
    ax_loss.set_title("Train loss", loc="left", fontsize=10, fontweight="bold")
    ax_loss.grid(alpha=0.25); sns.despine(ax=ax_loss)
    ax_loss.legend(loc="upper right", frameon=False)

    # ---- top-right: val Dice (lesion + anatomy) ----
    v_les = history.get("val_lesion_dice", [])
    v_ana = history.get("val_anatomy_dice", [])
    if v_les:
        ax_dice.plot(epoch[:len(v_les)], v_les, color=PALETTE[1],
                     linewidth=1.6, label="val lesion Dice")
        # mark peak
        valid = [(i, x) for i, x in enumerate(v_les) if x == x]
        if valid:
            best_i, best_v = max(valid, key=lambda t: t[1])
            ax_dice.scatter([epoch[best_i]], [best_v], color=PALETTE[1], s=42,
                            marker="*", zorder=6, edgecolor="black", linewidth=0.5)
            ax_dice.annotate(f"peak {best_v:.3f} @ ep {epoch[best_i]}",
                             xy=(epoch[best_i], best_v),
                             xytext=(8, -10), textcoords="offset points",
                             fontsize=8, fontweight="bold")
    if v_ana and any(x == x for x in v_ana):
        ax_dice.plot(epoch[:len(v_ana)], v_ana, color=PALETTE[2],
                     linewidth=1.2, linestyle="--", alpha=0.85,
                     label="val anatomy Dice")
    ax_dice.set_xlabel("Epoch")
    ax_dice.set_ylabel("Validation Dice", fontweight="bold")
    ax_dice.set_ylim(-0.02, 1.0)
    ax_dice.set_title("Validation Dice", loc="left", fontsize=10, fontweight="bold")
    ax_dice.axhline(0.5, color="0.7", linestyle=":", linewidth=0.8)
    ax_dice.grid(alpha=0.25); sns.despine(ax=ax_dice)
    ax_dice.legend(loc="lower right", frameon=False)

    # ---- bottom-left: LR + alpha ----
    lr = history.get("lr", [])
    alpha = history.get("alpha", [])
    if lr:
        ax_lr.plot(epoch[:len(lr)], lr, color=PALETTE[3], linewidth=1.4, label="learning rate")
        ax_lr.set_ylabel("Learning rate", color=PALETTE[3], fontweight="bold")
        ax_lr.tick_params(axis="y", colors=PALETTE[3])
        ax_lr.set_yscale("log")
    if alpha:
        ax_a = ax_lr.twinx()
        ax_a.plot(epoch[:len(alpha)], alpha, color=PALETTE[4], linewidth=1.4,
                  linestyle="--", label="curriculum α")
        ax_a.set_ylabel("Gate curriculum α", color=PALETTE[4], fontweight="bold")
        ax_a.tick_params(axis="y", colors=PALETTE[4])
        ax_a.set_ylim(-0.05, 1.05)
        ax_a.spines["top"].set_visible(False)
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_title("LR & curriculum α", loc="left", fontsize=10, fontweight="bold")
    ax_lr.grid(alpha=0.25); sns.despine(ax=ax_lr)

    # ---- bottom-right: per-term loss breakdown ----
    terms = {
        "train_les_dice_loss":  ("lesion Dice loss",      PALETTE[5]),
        "train_les_focal":      ("lesion Focal/Tversky",  PALETTE[6]),
        "train_ana_dice_loss":  ("anatomy Dice loss",     PALETTE[7]),
        "train_ana_cons":       ("anatomy constraint",    PALETTE[0]),
    }
    any_term = False
    for key, (lab, c) in terms.items():
        vals = history.get(key, [])
        vals = [v for v in vals if v == v]
        if vals and _np.std(vals) > 1e-6:
            any_term = True
            ax_terms.plot(epoch[:len(vals)], vals, label=lab, color=c, linewidth=1.2)
    ax_terms.set_xlabel("Epoch")
    ax_terms.set_ylabel("Loss term value", fontweight="bold")
    ax_terms.set_title("Per-term training loss breakdown", loc="left",
                       fontsize=10, fontweight="bold")
    if any_term:
        ax_terms.legend(loc="upper right", fontsize=7, frameon=False, ncol=1)
    ax_terms.grid(alpha=0.25); sns.despine(ax=ax_terms)

    fig.tight_layout()
    _save(fig, out_path, dpi=dpi)


def plot_lr_schedule(epochs: list, lrs: list, out_path: Path, dpi: int = DEFAULT_DPI) -> None:
    _rc()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(epochs, lrs, linewidth=1.4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Learning rate schedule")
    _save(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------------
# Prediction overlay
# ---------------------------------------------------------------------------
def overlay_pred_gt(image: np.ndarray, pred: np.ndarray, gt: np.ndarray,
                    out_path: Path, *,
                    title: str = "", dpi: int = DEFAULT_DPI) -> None:
    """Show image + GT (green) + pred (red) + intersection (yellow)."""
    _rc()
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    im = image.astype(np.float32)
    if im.max() > 1:
        im = im / im.max()
    axes[0].imshow(im, cmap="gray")
    axes[0].set_title("Image"); axes[0].axis("off")
    axes[1].imshow(im, cmap="gray")
    axes[1].imshow(np.where(gt > 0, 1.0, np.nan), cmap="Greens", alpha=0.6, vmin=0, vmax=1)
    axes[1].set_title("GT (green)"); axes[1].axis("off")
    axes[2].imshow(im, cmap="gray")
    axes[2].imshow(np.where(pred > 0, 1.0, np.nan), cmap="Reds", alpha=0.6, vmin=0, vmax=1)
    axes[2].set_title("Pred (red)"); axes[2].axis("off")
    # combined
    axes[3].imshow(im, cmap="gray")
    rgb = np.zeros((*im.shape, 4))
    rgb[..., 0] = (pred > 0) & ~(gt > 0)        # FP red
    rgb[..., 1] = (gt > 0) & ~(pred > 0)        # FN green
    inter = (pred > 0) & (gt > 0)
    rgb[..., 0] = np.where(inter, 1.0, rgb[..., 0])
    rgb[..., 1] = np.where(inter, 1.0, rgb[..., 1])
    rgb[..., 3] = ((pred > 0) | (gt > 0)).astype(np.float32) * 0.6
    axes[3].imshow(rgb)
    axes[3].set_title("FP=red, FN=green, TP=yellow"); axes[3].axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    _save(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------------
# FROC + ROC
# ---------------------------------------------------------------------------
def plot_froc(froc_points, out_path: Path, *,
              label: str = "AnchorBS-Lite", dpi: int = DEFAULT_DPI,
              other_curves: dict | None = None,
              fp_cap: float = 8.0) -> None:
    """froc_points: list of FROCPoint or list of (fp, sens) tuples"""
    _rc()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    curves = [(label, froc_points)]
    if other_curves:
        curves.extend(other_curves.items())
    for i, (name, pts) in enumerate(curves):
        x = [p.fp_per_image if hasattr(p, "fp_per_image") else p[0] for p in pts]
        y = [p.sensitivity if hasattr(p, "sensitivity") else p[1] for p in pts]
        marker = ["o", "s", "D", "^", "v", "p"][i % 6]
        ax.plot(x, y, marker=marker, label=name, linewidth=1.5,
                markersize=4.2, color=PALETTE[i % len(PALETTE)])
    ax.set_xlim(0, fp_cap)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positives per scan", fontweight="bold")
    ax.set_ylabel("Sensitivity (lesion-level)", fontweight="bold")
    ax.set_title("FROC curve", loc="left")
    ax.axvline(1.0, color="0.35", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.annotate("clinical operating point\n(1 FP / scan)",
                xy=(1.0, 0.05), xytext=(1.6, 0.10),
                fontsize=7.5, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.6))
    sns.despine(ax=ax)
    ax.legend(loc="lower right", frameon=False)
    _save(fig, out_path, dpi=dpi)


def plot_roc(fpr, tpr, auc: float, out_path: Path, *,
             label: str = "", dpi: int = DEFAULT_DPI) -> None:
    _rc()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, linewidth=1.6, label=f"{label} (AUC={auc:.3f})" if label else f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("ROC curve (pixel-wise)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    _save(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------------
# Box / violin for cross-method comparison
# ---------------------------------------------------------------------------
def boxplot_methods(per_method_values: dict, out_path: Path, *,
                    ylabel: str = "Dice",
                    dpi: int = DEFAULT_DPI,
                    show_swarm: bool = True) -> None:
    """Seaborn-style box + strip overlay (Nature-Methods style)."""
    _rc()
    import pandas as pd
    rows = []
    for m, vs in per_method_values.items():
        rows.extend([{"method": m, ylabel: v} for v in vs])
    df = pd.DataFrame(rows)
    if df.empty:
        return
    n = df.method.nunique()
    fig, ax = plt.subplots(figsize=(max(5, 1.0 * n + 2), 4.2))
    sns.boxplot(data=df, x="method", y=ylabel, ax=ax, width=0.55,
                palette=PALETTE[:n], linewidth=0.9, showfliers=False)
    if show_swarm and len(df) <= 800:
        sns.stripplot(data=df, x="method", y=ylabel, ax=ax,
                      color="0.25", size=2.2, alpha=0.55, jitter=0.15)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(f"{ylabel} per patient", loc="left")
    sns.despine(ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    _save(fig, out_path, dpi=dpi)


def barplot_with_ci(method_stats: dict, out_path: Path, *,
                    ylabel: str = "Dice",
                    dpi: int = DEFAULT_DPI) -> None:
    """method_stats: {method: {'mean':, 'ci95_lo':, 'ci95_hi':}}"""
    _rc()
    methods = list(method_stats.keys())
    means = [method_stats[m]["mean"] for m in methods]
    lo = [method_stats[m]["mean"] - method_stats[m]["ci95_lo"] for m in methods]
    hi = [method_stats[m]["ci95_hi"] - method_stats[m]["mean"] for m in methods]
    fig, ax = plt.subplots(figsize=(max(5, 1.0 * len(methods) + 2), 4.2))
    colors = PALETTE[: len(methods)]
    ax.bar(methods, means, yerr=[lo, hi], capsize=4, color=colors,
           edgecolor="black", linewidth=0.6, error_kw={"linewidth": 0.9, "ecolor": "0.25"})
    for x, m in zip(range(len(methods)), means):
        ax.text(x, m + max(hi) * 0.05, f"{m:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    ax.set_ylabel(ylabel + "  (mean ± 95% CI)", fontweight="bold")
    ax.set_title(f"{ylabel} across methods", loc="left")
    sns.despine(ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    _save(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm: np.ndarray, classes: list, out_path: Path,
                          dpi: int = DEFAULT_DPI, title: str = "Confusion matrix") -> None:
    _rc()
    fig, ax = plt.subplots(figsize=(max(4, len(classes) * 0.6), max(4, len(classes) * 0.6)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out_path, dpi=dpi)


# ---------------------------------------------------------------------------
# Calibration plot
# ---------------------------------------------------------------------------
def plot_reliability(probs, targets, out_path: Path, *,
                     n_bins: int = 15, dpi: int = DEFAULT_DPI) -> None:
    _rc()
    p = np.asarray(probs).flatten()
    t = np.asarray(targets).astype(np.float32).flatten()
    bins = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    accs = []
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1])
        accs.append(t[mask].mean() if mask.any() else np.nan)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(centers, accs, "o-", linewidth=1.5, markersize=5, label="Model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram (pixel-wise)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    _save(fig, out_path, dpi=dpi)
