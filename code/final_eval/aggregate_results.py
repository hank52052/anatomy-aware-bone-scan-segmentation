"""Aggregate eval_final.py outputs into the numbers, tables and statistics used in the manuscript.

Usage: python aggregate_results.py <eval_dir> <out_dir>
Writes <out_dir>/summary.json (single source of every number in the paper) and table CSVs.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

METHODS = [("ours_anat", "Ours (MAE + anatomy)"), ("ours_noanat", "Ours without anatomy loss"),
           ("bsciseg", "BSci-Seg (re-implemented)"), ("nnunet", "nnU-Net (2D)"),
           ("ours_ens", "Ours, two-model ensemble (secondary)"),
           ("ours_anat_kf", "Ours with kidney filter (secondary)")]
NOT_TESTED = {"ours_ens", "ours_anat_kf"}  # descriptive only, outside the Holm family
REFERENCE = "ours_anat"
N_BOOT = 5000
RNG = np.random.default_rng(20261020)


def load(eval_dir: Path, method: str) -> list[dict]:
    rows = []
    for fold in range(5):
        p = eval_dir / f"{method}_fold{fold}.json"
        if p.exists():
            rows += json.loads(p.read_text())["rows"]
    return rows


def dice(tp: int, fp: int, fn: int) -> float:
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else 0.0


def boot_mean_ci(values: np.ndarray) -> tuple[float, float]:
    idx = RNG.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(rows: list[dict]) -> dict:
    cv = [r for r in rows if r["group"] == "cv" and r["has_lesion_mask"]]
    pos_views = [r for r in cv if r["gt_area"] > 0]
    img_dsc = np.array([r["dice"] for r in pos_views], dtype=float)

    by_patient = defaultdict(list)
    for r in cv:
        by_patient[(r["source"], r["patient_id"])].append(r)
    pat = {}
    for key, views in by_patient.items():
        tp, fp, fn = (sum(v[k] for v in views) for k in ("tp", "fp", "fn"))
        pat[key] = {"fold": views[0]["fold"], "hospital": views[0]["hospital"], "gt": tp + fn > 0,
                    "dice": dice(tp, fp, fn) if tp + fn > 0 else None,
                    "fp_comp": sum(v["n_fp_components"] for v in views)}
    pos_pat = {k: v for k, v in pat.items() if v["gt"]}

    tp, fp, fn = (sum(r[k] for r in pos_views) for k in ("tp", "fp", "fn"))
    n_gt = sum(r["n_gt_lesions"] for r in pos_views)
    n_det = sum(r["n_gt_detected"] for r in pos_views)
    fold_dsc = [float(np.mean([r["dice"] for r in pos_views if r["fold"] == f])) for f in range(5)
                if any(r["fold"] == f for r in pos_views)]

    lesion_free = defaultdict(list)
    for r in rows:
        if r["lesion_free_patient"] and (r["group"] == "neg_new" or r["has_lesion_mask"]):
            lesion_free[(r["source"], r["patient_id"])].append(r)
    lf_comp = np.array([sum(v["n_fp_components"] for v in views) for views in lesion_free.values()], dtype=float)
    lf_area = np.array([sum(v["fp"] for v in views) for views in lesion_free.values()], dtype=float)
    lf_hosp = {k: views[0]["hospital"] for k, views in lesion_free.items()}

    out = {
        "n_cv_scored_patients": len(pat), "n_pos_patients": len(pos_pat), "n_pos_views": len(pos_views),
        "dsc_img_mean": float(img_dsc.mean()), "dsc_img_sd": float(img_dsc.std(ddof=1)),
        "dsc_img_median": float(np.median(img_dsc)),
        "dsc_img_ci": boot_mean_ci(img_dsc),
        "dsc_pat_mean": float(np.mean([v["dice"] for v in pos_pat.values()])),
        "dsc_pat_ci": boot_mean_ci(np.array([v["dice"] for v in pos_pat.values()])),
        "fold_dsc_img": fold_dsc, "fold_dsc_img_mean": float(np.mean(fold_dsc)),
        "fold_dsc_img_sd": float(np.std(fold_dsc, ddof=1)) if len(fold_dsc) > 1 else None,
        "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": dice(tp, fp, fn), "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "lesion_sensitivity": n_det / n_gt if n_gt else None, "n_gt_lesions": n_gt,
        "fp_comp_per_pos_scan": float(np.mean([v["fp_comp"] for v in pos_pat.values()])),
        "lesion_free_n": int(len(lf_comp)),
        "lesion_free_any_fp_pct": float((lf_comp > 0).mean() * 100) if len(lf_comp) else None,
        "lesion_free_fp_comp_mean": float(lf_comp.mean()) if len(lf_comp) else None,
        "lesion_free_fp_area_mean": float(lf_area.mean()) if len(lf_area) else None,
        "lesion_free_by_hospital": {
            h: {"n": int(sum(1 for k in lesion_free if lf_hosp[k] == h)),
                "any_fp_pct": float(np.mean([c > 0 for k, c in zip(lesion_free, lf_comp) if lf_hosp[k] == h]) * 100),
                "fp_comp_mean": float(np.mean([c for k, c in zip(lesion_free, lf_comp) if lf_hosp[k] == h]))}
            for h in ("old", "new") if any(lf_hosp[k] == h for k in lesion_free)},
        "by_hospital": {
            h: {"n_pos_views": int(sum(1 for r in pos_views if r["hospital"] == h)),
                "dsc_img_mean": float(np.mean([r["dice"] for r in pos_views if r["hospital"] == h])),
                "n_pos_patients": int(sum(1 for v in pos_pat.values() if v["hospital"] == h)),
                "dsc_pat_mean": float(np.mean([v["dice"] for v in pos_pat.values() if v["hospital"] == h]))}
            for h in ("old", "new") if any(r["hospital"] == h for r in pos_views)},
    }
    return out, {k: v["dice"] for k, v in pos_pat.items()}, img_dsc


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adjusted = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adjusted[i] = running
    return adjusted.tolist()


def main() -> None:
    global METHODS, NOT_TESTED
    if os.environ.get("AGG_METHODS"):  # e.g. "ours_anat:Ours two-stage,ours_1stage:Ours one-stage,..."
        METHODS = [tuple(item.split(":", 1)) for item in os.environ["AGG_METHODS"].split(",")]
        NOT_TESTED = set(filter(None, os.environ.get("AGG_NOT_TESTED", "").split(",")))
    eval_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, per_patient, per_image = {"methods": {}, "comparisons": {}}, {}, {}
    for key, label in METHODS:
        rows = load(eval_dir, key)
        if not rows:
            print(f"[skip] no results for {key}")
            continue
        folds_present = sorted({r["fold"] for r in rows})
        s, pp, pi = summarize(rows)
        s.update({"label": label, "folds_present": folds_present})
        summary["methods"][key] = s
        per_patient[key], per_image[key] = pp, pi
        print(f"{label:30s} folds={folds_present} DSC_img={s['dsc_img_mean']:.4f} DSC_pat={s['dsc_pat_mean']:.4f} "
              f"P={s['precision']:.3f} R={s['recall']:.3f} lesSens={s['lesion_sensitivity']:.3f} "
              f"FP/scan={s['fp_comp_per_pos_scan']:.2f} lesionfree anyFP={s['lesion_free_any_fp_pct']}")

    others = ([k for k, _ in METHODS if k != REFERENCE and k in per_patient and k not in NOT_TESTED]
              if REFERENCE in per_patient else [])
    pvals, diffs = [], {}
    for k in others:
        common = sorted(set(per_patient[REFERENCE]) & set(per_patient[k]))
        a = np.array([per_patient[REFERENCE][c] for c in common])
        b = np.array([per_patient[k][c] for c in common])
        d = a - b
        p = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue) if np.any(d != 0) else 1.0
        diffs[k] = {"n_patients": len(common), "mean_diff": float(d.mean()), "ci": boot_mean_ci(d),
                    "median_diff": float(np.median(d)), "p_wilcoxon": p,
                    "ref_better_pct": float((d > 0).mean() * 100)}
        pvals.append(p)
    for k, p_adj in zip(others, holm(pvals) if pvals else []):
        diffs[k]["p_holm"] = p_adj
    summary["comparisons"] = diffs

    # Secondary: paired comparison of false-positive component counts (lesion-free patients, and lesion-positive
    # patients), our model vs each other model. Descriptive Wilcoxon P values, no multiplicity adjustment.
    fp_cmp = {}
    fp_counts = {}
    for key, _ in METHODS:
        rows = load(eval_dir, key)
        lf, lp, gt = defaultdict(int), defaultdict(int), defaultdict(int)
        for r in rows:
            pid = (r["source"], r["patient_id"])
            if r["lesion_free_patient"] and (r["group"] == "neg_new" or r["has_lesion_mask"]):
                lf[pid] += r["n_fp_components"]
            elif r["group"] == "cv" and r["has_lesion_mask"]:
                lp[pid] += r["n_fp_components"]
                gt[pid] += r["gt_area"]
        lp = {k: v for k, v in lp.items() if gt[k] > 0}  # same lesion-positive set as Table 3
        fp_counts[key] = (lf, lp)
    if REFERENCE in fp_counts:
        for key, _ in METHODS:
            if key == REFERENCE or key not in fp_counts or not fp_counts[key][0]:
                continue
            entry = {}
            for idx, name in ((0, "lesion_free"), (1, "lesion_positive")):
                ref, oth = fp_counts[REFERENCE][idx], fp_counts[key][idx]
                common = sorted(set(ref) & set(oth))
                a = np.array([ref[c] for c in common], float)
                b = np.array([oth[c] for c in common], float)
                p = float(stats.wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0
                entry[name] = {"n": len(common), "ref_mean": float(a.mean()), "other_mean": float(b.mean()),
                               "mean_diff": float((a - b).mean()), "ci": boot_mean_ci(a - b), "p_wilcoxon": p}
            fp_cmp[key] = entry
            print(f"  FP ours_anat - {key}: lesion-free diff={entry['lesion_free']['mean_diff']:+.2f} "
                  f"p={entry['lesion_free']['p_wilcoxon']:.3g}; lesion-positive diff="
                  f"{entry['lesion_positive']['mean_diff']:+.2f} p={entry['lesion_positive']['p_wilcoxon']:.3g}")
    summary["fp_comparisons"] = fp_cmp
    for k, v in diffs.items():
        print(f"  ours_anat - {k}: diff={v['mean_diff']:+.4f} CI=({v['ci'][0]:+.4f},{v['ci'][1]:+.4f}) "
              f"p={v['p_wilcoxon']:.4g} holm={v.get('p_holm', float('nan')):.4g} n={v['n_patients']}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "per_patient_dsc.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        keys = sorted(set().union(*[set(v) for v in per_patient.values()]))
        w.writerow(["source", "patient_id"] + list(per_patient))
        for key in keys:
            w.writerow(list(key) + [f"{per_patient[m].get(key, float('nan')):.6f}" for m in per_patient])
    with open(out_dir / "lesion_free_fp.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "source", "patient_id", "hospital", "fp_components", "fp_pixels"])
        for key, _ in METHODS:
            groups = defaultdict(list)
            for r in load(eval_dir, key):
                if r["lesion_free_patient"] and (r["group"] == "neg_new" or r["has_lesion_mask"]):
                    groups[(r["source"], r["patient_id"])].append(r)
            for (src, pid), views in sorted(groups.items()):
                w.writerow([key, src, pid, views[0]["hospital"], sum(v["n_fp_components"] for v in views),
                            sum(v["fp"] for v in views)])
    print("wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()
