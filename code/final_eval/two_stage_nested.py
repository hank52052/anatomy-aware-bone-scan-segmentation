"""Nested two-stage pipeline for our model (rule fixed 2026-10-20 06:15 before any result was seen).

Base probability b: our anatomy model ("anat") or the mean of our anatomy and no-anatomy models ("ens").
Stage 2 (two_stage.py features and classifier): candidates are components with probability >= 0.10; a candidate is
kept when the classifier probability is >= q. Output "full" keeps the whole candidate, "core" keeps only its pixels
with probability >= 0.5 (a kept candidate without such pixels is kept whole). Two one-stage options (b at 0.5)
let the selection fall back to no second stage.

Outer fold k: for every inner fold j != k a classifier trained on the three folds outside {j, k} predicts fold j;
the configuration with the highest image-level Dice over the lesion-positive views of all j != k is chosen; a
classifier trained on the four folds other than k is then applied to fold k with that configuration.

Usage (from the repository root): CUDA_VISIBLE_DEVICES=1 python code/final_eval/two_stage_nested.py <out_dir>
Writes <out_dir>/ours_anat_fold{k}.json (rows of the nested pipeline, eval_final.py format),
       <out_dir>/fixed/<config>/ours_anat_fold{k}.json (each configuration with the outer classifier, for
       transparent reporting only) and <out_dir>/nested.json (choices and inner scores).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_final as ef  # noqa: E402
import two_stage as ts  # noqa: E402

BASES = ("anat", "ens")
QS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
MODES = ("full", "core")
CONFIGS = [(b, None, None) for b in BASES] + [(b, q, m) for b in BASES for q in QS for m in MODES]


def config_name(c):
    b, q, m = c
    return f"{b}_onestage" if q is None else f"{b}_q{q:.1f}_{m}"


def simplicity(c):
    b, q, m = c
    return (q is not None, 0.0 if q is None else round(abs(q - 0.5), 3), m == "core", b == "ens")


def extract(fold, lesion, negatives, splits):
    """Stage-1 outputs of one fold for both bases: candidate features, labels, and per-component pixel counts."""
    anat = ef.TwoViewPredictor("ours_anat", fold, keep_anatomy=True)
    noanat = ef.TwoViewPredictor("ours_noanat", fold)
    val = [lesion[i] for i in splits[fold][1]]
    neg = [r for r in negatives if ef.patient_id_hash(f"{r.source}_{r.patient_id}", 0) % 5 == fold]
    data = {b: {"views": [], "x": [], "y": [], "owner": [], "counts": []} for b in BASES}
    for group, records in (("cv", val), ("neg_new", neg)):
        ds = ef.BoneScanDataset(records, augment=False)
        for i, rec in enumerate(ds.records):
            item = ds[i]
            p_anat = anat(item["image"])
            labels_anat = anat.last_anatomy
            probs = {"anat": p_anat, "ens": (p_anat + noanat(item["image"])) / 2}
            img = item["image"].numpy()
            gt = item["lesion_mask"].numpy() > 0
            for b in BASES:
                p, d = probs[b], data[b]
                for v, view in enumerate(("ANT", "POST")):
                    other = ndimage.binary_dilation(p[1 - v] >= ts.T_CAND, structure=ef.STRUCT,
                                                    iterations=ts.XVIEW_DILATE)
                    lab, x = ts.component_features(p[v], labels_anat[v], img[v], other, v == 1)
                    n, g, core = len(x), gt[v], p[v] >= 0.5
                    counts = np.stack([np.bincount(lab[m], minlength=n + 1)[1:] for m in
                                       (np.ones_like(g), g, core, core & g)], axis=1)  # size, overlap, core, core-overlap
                    meta = {"group": group, "fold": fold, "source": rec.source, "patient_id": rec.patient_id,
                            "view": view, "hospital": "new" if rec.source in ef.NEW_HOSPITAL else "old",
                            "has_lesion_mask": bool(rec.has_lesion_mask), "lesion_free_patient":
                            rec.source in ef.NEG_SRC or rec.source.endswith("non-meta")}
                    d["views"].append({"meta": meta, "lab": lab.astype(np.uint16), "core": np.packbits(core),
                                       "gt": np.packbits(g), "shape": g.shape, "gt_area": int(g.sum()),
                                       "known": group == "neg_new" or rec.has_lesion_mask,
                                       "positive": group == "cv" and rec.has_lesion_mask and bool(g.any())})
                    d["x"].append(x)
                    d["y"].append(counts[:, 1] > 0)
                    d["counts"].append(counts)
                    d["owner"] += [len(d["views"]) - 1] * n
    for d in data.values():
        d["x"], d["y"], d["counts"] = np.concatenate(d["x"]), np.concatenate(d["y"]), np.concatenate(d["counts"])
        d["owner"] = np.array(d["owner"], dtype=int)
        d["known"] = np.array([d["views"][o]["known"] for o in d["owner"]], dtype=bool)
    del anat, noanat
    ef.torch.cuda.empty_cache()
    return data


def fit(data, folds, base):
    x = np.concatenate([data[f][base]["x"][data[f][base]["known"]] for f in folds])
    y = np.concatenate([data[f][base]["y"][data[f][base]["known"]] for f in folds])
    return HistGradientBoostingClassifier(random_state=0).fit(x, y)


def view_dices(d, prob_keep, config):
    """Image-level Dice of every lesion-positive view of one fold for one configuration (from component counts)."""
    _, q, m = config
    size, ov, core, core_ov = d["counts"].T
    if q is None:
        pred_c, tp_c = core, core_ov  # one stage: pixels >= 0.5
    else:
        keep = prob_keep >= q
        use_core = (m == "core") & (core > 0)
        pred_c = np.where(keep, np.where(use_core, core, size), 0)
        tp_c = np.where(keep, np.where(use_core, core_ov, ov), 0)
    n_views = len(d["views"])
    pred = np.bincount(d["owner"], weights=pred_c, minlength=n_views)
    tp = np.bincount(d["owner"], weights=tp_c, minlength=n_views)
    return [2 * tp[i] / (pred[i] + v["gt_area"]) for i, v in enumerate(d["views"]) if v["positive"]]


def final_rows(d, prob_keep, config):
    """Full eval_final metrics for every view of one fold under one configuration."""
    _, q, m = config
    size, _, core, _ = d["counts"].T
    rows = []
    for i, v in enumerate(d["views"]):
        shape = v["shape"]
        n = shape[0] * shape[1]
        gt = np.unpackbits(v["gt"])[:n].reshape(shape).astype(bool)
        core_mask = np.unpackbits(v["core"])[:n].reshape(shape).astype(bool)
        if q is None:
            mask = core_mask
        else:
            idx = np.where(d["owner"] == i)[0]
            kept = idx[prob_keep[idx] >= q] - idx.min() + 1 if len(idx) else np.array([], int)
            whole = np.isin(v["lab"], kept)
            if m == "core":
                kept_with_core = kept[core[idx.min() + kept - 1] > 0] if len(kept) else kept
                whole_only = np.isin(v["lab"], np.setdiff1d(kept, kept_with_core))
                mask = (whole & core_mask) | whole_only
            else:
                mask = whole
        r = ef.view_metrics(mask, gt)
        r.update(v["meta"])
        r["dice"] = ef.dice(r["tp"], r["fp"], r["fn"])
        rows.append(r)
    return rows


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    df = ef.load_manifest(ef.DATA_ROOT)
    recs = ef.assemble_patient_records(df, ef.DATA_ROOT)
    lesion = [r for r in recs if r.source in ef.LESION_SRC]
    negatives = [r for r in recs if r.source in ef.NEG_SRC]
    splits = ef.k_fold_split(lesion, k=5, seed=0)
    data = {k: extract(k, lesion, negatives, splits) for k in range(5)}
    print("extraction done", flush=True)

    order = sorted(CONFIGS, key=simplicity)
    report = {"configs": [config_name(c) for c in CONFIGS], "folds": {}}
    for k in range(5):
        scores = {c: [] for c in CONFIGS}
        for j in (f for f in range(5) if f != k):
            train = [f for f in range(5) if f not in (j, k)]
            for b in BASES:
                d = data[j][b]
                p = fit(data, train, b).predict_proba(d["x"])[:, 1] if len(d["x"]) else np.zeros(0)
                for c in CONFIGS:
                    if c[0] == b:
                        scores[c] += view_dices(d, p, c)
        means = {c: float(np.mean(v)) for c, v in scores.items()}
        best = max(means.values())
        chosen = next(c for c in order if means[c] == best)

        outer = {b: (fit(data, [f for f in range(5) if f != k], b).predict_proba(data[k][b]["x"])[:, 1]
                     if len(data[k][b]["x"]) else np.zeros(0)) for b in BASES}
        rows = final_rows(data[k][chosen[0]], outer[chosen[0]], chosen)
        (out / f"ours_anat_fold{k}.json").write_text(json.dumps(
            {"method": "ours_nested_two_stage", "fold": k, "config": config_name(chosen), "rows": rows}))
        for c in CONFIGS:  # every configuration with the outer classifier, for transparent reporting only
            (out / "fixed" / config_name(c)).mkdir(parents=True, exist_ok=True)
            (out / "fixed" / config_name(c) / f"ours_anat_fold{k}.json").write_text(json.dumps(
                {"method": config_name(c), "fold": k, "rows": final_rows(data[k][c[0]], outer[c[0]], c)}))
        pos = [r["dice"] for r in rows if r["group"] == "cv" and r["has_lesion_mask"] and r["gt_area"] > 0]
        report["folds"][str(k)] = {"chosen": config_name(chosen), "inner_scores": {config_name(c): v for c, v in means.items()},
                                   "outer_dsc_img": float(np.mean(pos))}
        print(f"[fold {k}] chosen {config_name(chosen)} (inner {best:.4f}) -> outer DSC_img {np.mean(pos):.4f}", flush=True)
    (out / "nested.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
