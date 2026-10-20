"""Secondary analysis: post-processing chosen by cross-validation (rule fixed 2026-10-20 05:15, before any result
was seen).

Configurations (eval_final.py PP_CONFIGS): threshold t, minimum component size s, minimum component peak
probability h, and skeleton filter a (only for models with an anatomy branch). For every method and fold k, the
configuration with the highest mean image-level Dice over the lesion-positive views of the other four folds is
applied to fold k. Ties go to the simplest configuration: t closest to 0.5, then smaller s, h, and a.

Usage: python postproc_cv.py <pp_dir> <out_dir> [<primary_eval_dir>]
Reads  <pp_dir>/{method}_fold{k}_pp.npz   (eval_final.py with PP_GRID=1)
Writes <out_dir>/{method}_fold{k}.json     (rows at the chosen configuration, eval_final.py format)
       <out_dir>/pp_choices.json          (chosen configuration per fold and the scores they were chosen on)
With <primary_eval_dir>, checks that t=0.5 without filters reproduces the primary rows.
"""
import json
import sys
from pathlib import Path

import numpy as np

METHODS = ["ours_anat", "ours_noanat", "bsciseg", "nnunet", "ours_ens", "ours_anat_kf"]
N_FOLDS = 5


def load_fold(path: Path) -> dict:
    z = np.load(path)
    fields = [str(f) for f in z["fields"]]
    return {"metrics": z["metrics"].astype(np.int64), "configs": [tuple(c) for c in z["configs"]],
            "fields": fields, "meta": json.loads(str(z["meta"])), "has_skeleton": bool(z["has_skeleton"]),
            "checkpoint": str(z["checkpoint"])}


def view_dice(metrics: np.ndarray, fields: list[str]) -> np.ndarray:
    """(n_views, n_configs) Dice per view and configuration."""
    tp, fp, fn = (metrics[..., fields.index(k)] for k in ("tp", "fp", "fn"))
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)


def positive_views(meta: list[dict], metrics: np.ndarray, fields: list[str]) -> np.ndarray:
    gt_area = metrics[:, 0, fields.index("gt_area")]
    return np.array([m["group"] == "cv" and m["has_lesion_mask"] for m in meta]) & (gt_area > 0)


def simplicity(config: tuple) -> tuple:
    t, s, h, a = config
    return (round(abs(t - 0.5), 4), s, h, a, t)


def choose(scores_by_fold: dict, fold: int, allowed: np.ndarray, order: list[int]) -> tuple[int, np.ndarray]:
    """scores_by_fold[f] = (sum of Dice per config, n views). Uses every fold except `fold`."""
    total = sum(v[0] for f, v in scores_by_fold.items() if f != fold)
    count = sum(v[1] for f, v in scores_by_fold.items() if f != fold)
    mean = np.where(allowed, total / count, -np.inf)
    best = mean.max()
    return next(i for i in order if mean[i] == best), mean


def main() -> None:
    pp_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    primary = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"rule": "per fold k: configuration maximising mean image-level Dice of lesion-positive views in the "
                      "other four folds; ties to the simplest configuration", "methods": {}}
    for method in METHODS:
        paths = {k: pp_dir / f"{method}_fold{k}_pp.npz" for k in range(N_FOLDS)}
        if not all(p.exists() for p in paths.values()):
            print(f"[skip] {method}: folds {[k for k, p in paths.items() if p.exists()]}")
            continue
        data = {k: load_fold(p) for k, p in paths.items()}
        configs, fields = data[0]["configs"], data[0]["fields"]
        allowed = np.array([data[0]["has_skeleton"] or c[3] == 0 for c in configs])
        order = sorted(range(len(configs)), key=lambda i: simplicity(configs[i]))
        scores = {}
        for k, d in data.items():
            pos = positive_views(d["meta"], d["metrics"], fields)
            scores[k] = (view_dice(d["metrics"][pos], fields).sum(axis=0), int(pos.sum()))

        for k in data:  # leakage self-test: fold k's own results must not change its choice
            before, _ = choose(scores, k, allowed, order)
            rng = np.random.default_rng(k)
            perturbed = dict(scores)
            perturbed[k] = (rng.random(len(configs)) * scores[k][1], scores[k][1])
            assert choose(perturbed, k, allowed, order)[0] == before, f"{method} fold {k}: leakage"

        chosen = {}
        for k, d in data.items():
            i, mean = choose(scores, k, allowed, order)
            chosen[k] = configs[i]
            rows = []
            for meta, m in zip(d["meta"], d["metrics"][:, i]):
                r = dict(meta)
                r.update({f: int(v) for f, v in zip(fields, m)})
                r["dice"] = (2 * r["tp"] / (2 * r["tp"] + r["fp"] + r["fn"])) if r["tp"] + r["fn"] > 0 else None
                rows.append(r)
            out = {"method": method, "fold": k, "postprocessing": dict(zip("tsha", configs[i])),
                   "rule": "cv_other_folds", "checkpoint": d["checkpoint"], "rows": rows}
            (out_dir / f"{method}_fold{k}.json").write_text(json.dumps(out))

            if primary is not None and (primary / f"{method}_fold{k}.json").exists():
                base = configs.index((0.5, 0.0, 0.0, 0.0))
                key = lambda r: (r["source"], r["patient_id"], r["view"])
                ref = {key(r): r for r in json.loads((primary / f"{method}_fold{k}.json").read_text())["rows"]}
                changed = sum(any(int(m[fields.index(f)]) != ref[key(meta)][f] for f in ("tp", "fp", "fn", "n_fp_components"))
                              for meta, m in zip(d["meta"], d["metrics"][:, base]))
                if changed:
                    print(f"  [check] {method} fold {k}: {changed} views differ from the primary rows at t=0.5")
        record["methods"][method] = {"choices": {str(k): dict(zip("tsha", chosen[k])) for k in sorted(chosen)}}
        print(f"{method:14s} " + "  ".join(f"f{k}: t={c[0]:.2f} s={c[1]:.0f} h={c[2]:.1f} a={c[3]:.0f}"
                                            for k, c in sorted(chosen.items())))
    (out_dir / "pp_choices.json").write_text(json.dumps(record, indent=2))
    print("leakage self-test passed; wrote", out_dir)


if __name__ == "__main__":
    main()
