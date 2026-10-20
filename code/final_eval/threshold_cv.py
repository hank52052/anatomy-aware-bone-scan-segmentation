"""Secondary analysis: decision threshold chosen by cross-validation (rule fixed 2026-10-20 05:10, before any
result was seen).

For every method and every fold k, the threshold in the grid (0.10 to 0.90, step 0.05) that maximises the mean
image-level Dice over the lesion-positive views of the other four folds (each view predicted by the model of its
own fold) is applied to fold k. Fold k never contributes to its own threshold. Ties go to the threshold closest
to 0.5. Lesion-free patients use the threshold of the fold they belong to.

Usage: python threshold_cv.py <thr_dir> <out_dir>
Reads  <thr_dir>/{method}_fold{k}_thr.json  (eval_final.py with THRESH_GRID=1)
Writes <out_dir>/{method}_fold{k}.json       (rows at the selected threshold, same format as eval_final.py,
                                             so aggregate_results.py computes every metric the same way)
       <out_dir>/cv_thresholds.json         (selected thresholds and the scores they were chosen on)
"""
import json
import sys
from pathlib import Path

import numpy as np

METHODS = ["ours_anat", "ours_noanat", "bsciseg", "nnunet", "ours_ens", "ours_anat_kf"]
N_FOLDS = 5


def positive_view_dice(rows: list[dict]) -> list[float]:
    return [r["dice"] for r in rows if r["group"] == "cv" and r["has_lesion_mask"] and r["gt_area"] > 0]


def select_threshold(curves: dict, fold: int, grid: list[str]) -> tuple[str, dict]:
    """curves[f][t] = image-level Dice values of fold f at threshold t. Uses every fold except `fold`."""
    scores = {t: float(np.mean([d for f, c in curves.items() if f != fold for d in c[t]])) for t in grid}
    best = max(scores.values())
    tied = [t for t in grid if scores[t] == best]
    return min(tied, key=lambda t: (abs(float(t) - 0.5), float(t))), scores


def leakage_selftest(curves: dict, grid: list[str]) -> None:
    """Changing fold k's own results must not change fold k's threshold."""
    for k in curves:
        before, _ = select_threshold(curves, k, grid)
        rng = np.random.default_rng(k)
        perturbed = dict(curves)
        perturbed[k] = {t: list(rng.random(len(v))) for t, v in curves[k].items()}
        after, _ = select_threshold(perturbed, k, grid)
        assert before == after, f"fold {k}: threshold depends on its own data"


def main() -> None:
    thr_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"rule": "per fold k: grid threshold maximising mean image-level Dice of lesion-positive views in "
                      "the other four folds; ties to the value closest to 0.5", "grid": None, "methods": {}}
    for method in METHODS:
        data = {k: json.loads(p.read_text()) for k in range(N_FOLDS)
                if (p := thr_dir / f"{method}_fold{k}_thr.json").exists()}
        if len(data) < N_FOLDS:
            print(f"[skip] {method}: only folds {sorted(data)}")
            continue
        grid = [f"{t:.2f}" for t in data[0]["grid"]]
        record["grid"] = [float(t) for t in grid]
        curves = {k: {t: positive_view_dice(d["rows_by_t"][t]) for t in grid} for k, d in data.items()}
        leakage_selftest(curves, grid)
        chosen, scores = {}, {}
        for k, d in data.items():
            t, scores[k] = select_threshold(curves, k, grid)
            chosen[k] = float(t)
            out = {"method": method, "fold": k, "threshold": float(t), "threshold_rule": "cv_other_folds",
                   "checkpoint": d["checkpoint"], "rows": d["rows_by_t"][t]}
            (out_dir / f"{method}_fold{k}.json").write_text(json.dumps(out))
        record["methods"][method] = {"thresholds": {str(k): chosen[k] for k in sorted(chosen)},
                                     "selection_scores": {str(k): scores[k] for k in sorted(scores)}}
        print(f"{method:14s} thresholds by fold {[chosen[k] for k in sorted(chosen)]}")
    (out_dir / "cv_thresholds.json").write_text(json.dumps(record, indent=2))
    print("leakage self-test passed; wrote", out_dir)


if __name__ == "__main__":
    main()
