"""Patient-level detection AUC (rule fixed 2026-10-20 06:10 before computing).

Score of a patient = sum of the lesion probability over both views (expected lesion area). Positives: the
lesion-positive cross-validation patients; negatives: the lesion-free patients (Hospital A without metastases in
cross-validation, Hospital B without metastases scored by the model of their hash fold). Our model = mean of the
probabilities of our networks with and without the anatomy task. 95% CIs from 5,000 bootstrap resamples of
patients; differences (ours minus other) from paired resamples.

Usage (from the repository root): python code/final_eval/patient_auc.py <out_json>
"""
import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

N_BOOT = 5000
RNG = np.random.default_rng(20261020)
SINGLE = ("ours_anat", "ours_noanat", "bsciseg", "nnunet")


def patient_scores(method):
    scores, label = defaultdict(float), {}
    for k in range(5):
        prob = np.load(f"runs/final_eval_thr/{method}_fold{k}_prob.npz")
        rows = json.load(open(f"runs/final_eval/{method}_fold{k}.json"))["rows"]
        gt_area = defaultdict(int)
        for r in rows:
            gt_area[(r["source"], r["patient_id"])] += r["gt_area"]
        for r in rows:
            pid = (r["source"], r["patient_id"])
            if r["group"] == "cv" and r["has_lesion_mask"] and gt_area[pid] > 0:
                label[pid] = 1
            elif r["lesion_free_patient"] and (r["group"] == "neg_new" or r["has_lesion_mask"]):
                label[pid] = 0
            else:
                continue
            scores[pid] += float(prob["|".join((r["source"], r["patient_id"], r["view"]))].astype(np.float32).sum())
    return scores, label


def bootstrap(y, s, others=None):
    n = len(y)
    aucs, diffs = [], {k: [] for k in (others or {})}
    for _ in range(N_BOOT):
        i = RNG.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        a = roc_auc_score(y[i], s[i])
        aucs.append(a)
        for k, so in (others or {}).items():
            diffs[k].append(a - roc_auc_score(y[i], so[i]))
    return aucs, diffs


def main():
    scores, labels = {}, None
    for m in SINGLE:
        scores[m], lab = patient_scores(m)
        labels = lab if labels is None else labels
        assert lab == labels, f"{m}: patient set differs"
    pids = sorted(labels)
    y = np.array([labels[p] for p in pids])
    s = {m: np.array([scores[m][p] for p in pids]) for m in SINGLE}
    s["ours_ens"] = (s["ours_anat"] + s["ours_noanat"]) / 2  # mean probability map -> mean of sums
    out = {"n_positive": int(y.sum()), "n_negative": int((1 - y).sum()), "methods": {}, "differences": {}}
    for m, sm in s.items():
        aucs, _ = bootstrap(y, sm)
        fpr, tpr, _ = roc_curve(y, sm)
        out["methods"][m] = {"auc": float(roc_auc_score(y, sm)), "ci": [float(np.percentile(aucs, 2.5)),
                                                                         float(np.percentile(aucs, 97.5))],
                             "roc": {"fpr": fpr.round(5).tolist(), "tpr": tpr.round(5).tolist()}}
    _, diffs = bootstrap(y, s["ours_ens"], {m: s[m] for m in SINGLE})
    for m, d in diffs.items():
        d = np.array(d)
        p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        out["differences"][m] = {"diff": float(roc_auc_score(y, s["ours_ens"]) - roc_auc_score(y, s[m])),
                                 "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_boot": p}
    json.dump(out, open(sys.argv[1], "w"), indent=2)
    for m, v in out["methods"].items():
        print(f"{m:12s} AUC {v['auc']:.3f} ({v['ci'][0]:.3f} to {v['ci'][1]:.3f})")
    for m, v in out["differences"].items():
        print(f"  ours_ens - {m}: {v['diff']:+.3f} ({v['ci'][0]:+.3f} to {v['ci'][1]:+.3f}) p={v['p_boot']:.3f}")
    print("positives", out["n_positive"], "negatives", out["n_negative"])


if __name__ == "__main__":
    main()
