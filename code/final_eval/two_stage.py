"""Two-stage version of our model (added after the first comparison; rule fixed 2026-10-20 05:40).

Stage 1: our model (MAE + anatomy task); candidates are 8-connected components with lesion probability >= 0.10.
Stage 2: a gradient-boosting classifier keeps a candidate when its predicted probability of touching an
annotated lesion is >= 0.5. For fold k the classifier is trained only on the candidates of the other four folds,
each predicted by the model of its own fold. A second classifier without the anatomy features shows what the
anatomy branch contributes.

Usage (from the repository root): CUDA_VISIBLE_DEVICES=1 python code/final_eval/two_stage.py <out_dir>
Writes <out_dir>/full/ours_anat_fold{k}.json and <out_dir>/no_anatomy_features/ours_anat_fold{k}.json (rows in the
eval_final.py format) and <out_dir>/two_stage.json (classifier AUC per fold, candidate counts).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_final as ef  # noqa: E402

T_CAND, KEEP, N_ANAT, RING, XVIEW_DILATE = 0.10, 0.5, 11, 3, 5
ANATOMY_FEATURES = [f"anat_{c}" for c in range(N_ANAT)]
FEATURES = (["log_area", "peak_prob", "mean_prob", "frac_ge_050", "frac_ge_090"] + ANATOMY_FEATURES
            + ["cy", "cx", "bbox_h", "bbox_w", "fill", "img_mean", "img_max", "ring_mean", "contrast",
               "view_post", "xview_support"])


def component_features(prob, anat, img, other_candidates, view_post):
    """Candidate components of one view and one feature row per component."""
    h, w = prob.shape
    lab, n = ndimage.label(prob >= T_CAND, structure=ef.STRUCT)
    if n == 0:
        return lab, np.zeros((0, len(FEATURES)))
    ids = np.arange(1, n + 1)
    ones = np.ones_like(prob)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(float)
    cy, cx = np.array(ndimage.center_of_mass(ones, lab, ids)).T
    objs = ndimage.find_objects(lab)
    bh = np.array([o[0].stop - o[0].start for o in objs], float)
    bw = np.array([o[1].stop - o[1].start for o in objs], float)
    ring_mean = np.empty(n)
    for i, o in enumerate(objs):
        sl = tuple(slice(max(s.start - RING, 0), s.stop + RING) for s in o)
        comp = lab[sl] == i + 1
        ring = ndimage.binary_dilation(comp, structure=ef.STRUCT, iterations=RING) & ~comp
        ring_mean[i] = img[sl][ring].mean() if ring.any() else 0.0
    img_mean = np.asarray(ndimage.mean(img, lab, ids))
    columns = [np.log(area), np.asarray(ndimage.maximum(prob, lab, ids)), np.asarray(ndimage.mean(prob, lab, ids)),
               np.asarray(ndimage.mean((prob >= 0.5).astype(float), lab, ids)),
               np.asarray(ndimage.mean((prob >= 0.9).astype(float), lab, ids))]
    columns += [np.asarray(ndimage.mean((anat == c).astype(float), lab, ids)) for c in range(N_ANAT)]
    columns += [cy / h, cx / w, bh / h, bw / w, area / (bh * bw), img_mean,
                np.asarray(ndimage.maximum(img, lab, ids)), ring_mean, img_mean / (ring_mean + 1e-6),
                np.full(n, float(view_post)), np.asarray(ndimage.mean(other_candidates.astype(float), lab, ids))]
    return lab, np.column_stack(columns)


def extract(fold, lesion, negatives, splits):
    """Run stage 1 on the scored patients of one fold; return per-view records and the candidate table."""
    predictor = ef.TwoViewPredictor("ours_anat", fold, keep_anatomy=True)
    val = [lesion[i] for i in splits[fold][1]]
    neg = [r for r in negatives if ef.patient_id_hash(f"{r.source}_{r.patient_id}", 0) % 5 == fold]
    views, feats, labels, owners = [], [], [], []
    for group, records in (("cv", val), ("neg_new", neg)):
        ds = ef.BoneScanDataset(records, augment=False)
        for i, rec in enumerate(ds.records):
            item = ds[i]
            prob = predictor(item["image"])
            anat = predictor.last_anatomy
            img = item["image"].numpy()
            gt = item["lesion_mask"].numpy() > 0
            for v, view in enumerate(("ANT", "POST")):
                other = ndimage.binary_dilation(prob[1 - v] >= T_CAND, structure=ef.STRUCT, iterations=XVIEW_DILATE)
                lab, f = component_features(prob[v], anat[v], img[v], other, v == 1)
                touches = np.bincount(lab[gt[v]], minlength=lab.max() + 1)[1:] > 0
                meta = {"group": group, "fold": fold, "source": rec.source, "patient_id": rec.patient_id,
                        "view": view, "hospital": "new" if rec.source in ef.NEW_HOSPITAL else "old",
                        "has_lesion_mask": bool(rec.has_lesion_mask), "lesion_free_patient":
                        rec.source in ef.NEG_SRC or rec.source.endswith("non-meta")}
                known = group == "neg_new" or rec.has_lesion_mask  # label of every candidate is known
                views.append({"meta": meta, "lab": lab.astype(np.uint16), "gt": gt[v], "known": known})
                feats.append(f)
                labels.append(touches)
                owners += [(len(views) - 1, c + 1) for c in range(len(f))]
    del predictor
    ef.torch.cuda.empty_cache()
    return views, np.concatenate(feats), np.concatenate(labels), owners


def main():
    out = Path(sys.argv[1])
    df = ef.load_manifest(ef.DATA_ROOT)
    recs = ef.assemble_patient_records(df, ef.DATA_ROOT)
    lesion = [r for r in recs if r.source in ef.LESION_SRC]
    negatives = [r for r in recs if r.source in ef.NEG_SRC]
    splits = ef.k_fold_split(lesion, k=5, seed=0)

    data = {k: extract(k, lesion, negatives, splits) for k in range(5)}
    report = {"t_candidate": T_CAND, "keep": KEEP, "features": FEATURES, "folds": {}}
    for variant, drop in (("full", []), ("no_anatomy_features", ANATOMY_FEATURES)):
        cols = [i for i, name in enumerate(FEATURES) if name not in drop]
        (out / variant).mkdir(parents=True, exist_ok=True)
        for k in range(5):
            x_tr, y_tr = [], []
            for j in range(5):
                if j == k:
                    continue
                views, x, y, owners = data[j]
                known = np.array([views[o[0]]["known"] for o in owners], dtype=bool)
                x_tr.append(x[known][:, cols])
                y_tr.append(y[known])
            clf = HistGradientBoostingClassifier(random_state=0).fit(np.concatenate(x_tr), np.concatenate(y_tr))
            views, x, y, owners = data[k]
            p = clf.predict_proba(x[:, cols])[:, 1] if len(x) else np.zeros(0)
            kept = {}
            for (vi, cid), pk in zip(owners, p):
                if pk >= KEEP:
                    kept.setdefault(vi, []).append(cid)
            rows = []
            for vi, view in enumerate(views):
                mask = np.isin(view["lab"], kept.get(vi, [])) if vi in kept else np.zeros_like(view["gt"])
                m = ef.view_metrics(mask, view["gt"])
                m.update(view["meta"])
                m["dice"] = ef.dice(m["tp"], m["fp"], m["fn"])
                rows.append(m)
            (out / variant / f"ours_anat_fold{k}.json").write_text(json.dumps(
                {"method": f"ours_anat_two_stage_{variant}", "fold": k, "rows": rows}))
            known = np.array([views[o[0]]["known"] for o in owners], dtype=bool)
            auc = float(roc_auc_score(y[known], p[known])) if len(set(y[known])) == 2 else None
            report["folds"].setdefault(variant, {})[str(k)] = {
                "n_candidates": int(len(x)), "n_kept": int((p >= KEEP).sum()), "auc": auc,
                "n_train": int(sum(len(a) for a in y_tr))}
            pos = [r["dice"] for r in rows if r["group"] == "cv" and r["has_lesion_mask"] and r["gt_area"] > 0]
            print(f"[{variant} fold{k}] candidates={len(x)} kept={(p >= KEEP).sum()} "
                  f"AUC={auc if auc is None else round(auc, 3)} "
                  f"DSC_img={np.mean(pos):.4f}", flush=True)
    (out / "two_stage.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
