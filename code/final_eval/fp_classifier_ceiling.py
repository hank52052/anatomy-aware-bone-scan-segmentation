"""Upper bound for a second-stage false-positive classifier (exploratory, not in the manuscript).

A perfect classifier keeps every predicted component that touches an annotated lesion and removes every other
component. It cannot add missed lesions. Reported: image-level DSC now, with perfect removal at threshold 0.5
(the primary masks), and with perfect removal of candidates generated at threshold 0.10.

Usage (from the repository root): python code/final_eval/fp_classifier_ceiling.py
"""
import json
import os

import numpy as np
from scipy import ndimage

STRUCT = np.ones((3, 3), dtype=bool)


def dice(mask: np.ndarray, gt: np.ndarray) -> float:
    return 2 * (mask & gt).sum() / (mask.sum() + gt.sum())


def keep_touching(mask: np.ndarray, gt: np.ndarray) -> np.ndarray:
    lab, _ = ndimage.label(mask, STRUCT)
    ids = np.unique(lab[gt & mask])
    return np.isin(lab, ids[ids > 0])


def ceiling(method: str) -> tuple[list[int], dict]:
    out = {"now": [], "perfect_at_050": [], "perfect_at_010": []}
    folds = [f for f in (range(3) if method == "nnunet" else range(5)) if os.path.exists(f"runs/final_eval_thr/{method}_fold{f}_prob.npz")
             and os.path.exists(f"runs/final_eval/{method}_fold{f}.json")]
    for f in folds:
        prob = np.load(f"runs/final_eval_thr/{method}_fold{f}_prob.npz")
        pred = np.load(f"runs/final_eval/{method}_fold{f}_pred.npz")
        for r in json.load(open(f"runs/final_eval/{method}_fold{f}.json"))["rows"]:
            if not (r["group"] == "cv" and r["has_lesion_mask"] and r["gt_area"] > 0):
                continue
            key = "|".join((r["source"], r["patient_id"], r["view"]))
            shape = tuple(pred[key + "|shape"])
            n = shape[0] * shape[1]
            gt = np.unpackbits(prob[key + "|gt"])[:n].reshape(shape).astype(bool)
            m50 = np.unpackbits(pred[key])[:n].reshape(shape).astype(bool)  # exact primary mask at 0.5
            m10 = prob[key].astype(np.float32) >= 0.10
            out["now"].append(dice(m50, gt))
            out["perfect_at_050"].append(dice(keep_touching(m50, gt), gt))
            out["perfect_at_010"].append(dice(keep_touching(m10, gt), gt))
    return folds, {k: float(np.mean(v)) for k, v in out.items()}


if __name__ == "__main__":
    for method in ("ours_anat", "ours_noanat", "bsciseg", "nnunet"):
        folds, r = ceiling(method)
        print(f"{method:12s} folds {folds}: now {r['now']:.4f} | perfect classifier at 0.5 {r['perfect_at_050']:.4f} "
              f"| perfect classifier on 0.10 candidates {r['perfect_at_010']:.4f}")
