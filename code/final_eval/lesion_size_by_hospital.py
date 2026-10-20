"""Exploratory (post hoc) analysis of the gap between hospitals: lesion size, number, and relative intensity of the
annotated lesions, and how detection and overlap depend on lesion size in each hospital.

Lesions are 8-connected components of the reference masks of the lesion-positive views in cross-validation (the same
views as the image-level DSC). Predicted masks and per-view DSC come from the fixed-threshold evaluation
(runs/final_eval). Hospital comparisons use per-patient summaries (Mann-Whitney U), so that lesions of one patient do
not count as independent observations. The hospital effect on per-view DSC after adjustment for the total lesion
area of the view is the coefficient of an ordinary least-squares fit DSC ~ log10(area) + TTMHH, with a 95% CI
from 5,000 bootstrap resamples of patients. Hospitals are keyed "A" (China Medical University Hospital, CMUH) and
"B" (Tungs' Taichung MetroHarbor Hospital, TTMHH) in the output.
Usage (from the repository root): python code/final_eval/lesion_size_by_hospital.py <out_json>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_final as ev  # noqa: E402

EVAL_DIR = ev.BONESCAN / "runs" / "final_eval"
METHODS = ("ours_ens", "ours_anat", "ours_noanat", "bsciseg", "nnunet")
SIZE_BINS = [1, 10, 32, 100, 316, 10 ** 6]  # half-decade bins of lesion area (pixels)
SIZE_LABELS = ["1-9", "10-31", "32-99", "100-315", ">=316"]
N_BOOT = 5000
RNG = np.random.default_rng(20261020)


def unpack(npz, key):
    shape = tuple(npz[f"{key}|shape"])
    return np.unpackbits(npz[key])[: shape[0] * shape[1]].reshape(shape).astype(bool)


def summary(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"n": int(v.size), "median": float(np.median(v)), "q1": float(np.percentile(v, 25)),
            "q3": float(np.percentile(v, 75)), "mean": float(v.mean())}


def ols_hospital(rows) -> dict:
    """DSC ~ 1 + log10(total lesion area) + TTMHH indicator; CI of the hospital coefficient by patient bootstrap."""
    def fit(rr):
        x = np.column_stack([np.ones(len(rr)), np.log10([r["area"] for r in rr]), [r["hosp"] == "B" for r in rr]])
        return np.linalg.lstsq(x, np.array([r["dsc"] for r in rr]), rcond=None)[0]

    beta = fit(rows)
    by_pat = defaultdict(list)
    for r in rows:
        by_pat[r["pid"]].append(r)
    pids = list(by_pat)
    boots = []
    for _ in range(N_BOOT):
        sample = [r for i in RNG.integers(0, len(pids), len(pids)) for r in by_pat[pids[i]]]
        boots.append(fit(sample))
    boots = np.array(boots)
    raw_gap = (np.mean([r["dsc"] for r in rows if r["hosp"] == "B"])
               - np.mean([r["dsc"] for r in rows if r["hosp"] == "A"]))
    return {"hospital_b_coef": float(beta[2]), "hospital_b_ci": [float(np.percentile(boots[:, 2], 2.5)),
                                                               float(np.percentile(boots[:, 2], 97.5))],
            "log10_area_coef": float(beta[1]), "log10_area_ci": [float(np.percentile(boots[:, 1], 2.5)),
                                                                 float(np.percentile(boots[:, 1], 97.5))],
            "unadjusted_gap": float(raw_gap)}


def main() -> None:
    df = ev.load_manifest(ev.DATA_ROOT)
    recs = ev.assemble_patient_records(df, ev.DATA_ROOT)
    lesion = [r for r in recs if r.source in ev.LESION_SRC and r.has_lesion_mask]
    ds = ev.BoneScanDataset(lesion, augment=False)

    lesions, views = [], []  # one row per reference lesion / per lesion-positive view
    for i, rec in enumerate(ds.records):
        item = ds[i]
        gt = item["lesion_mask"].numpy() > 0
        img = item["image"].numpy()
        hosp = "B" if rec.source in ev.NEW_HOSPITAL else "A"
        pid = f"{rec.source}|{rec.patient_id}"
        for v, view in enumerate(("ANT", "POST")):
            if not gt[v].any():
                continue
            lab, n = ndimage.label(gt[v], structure=ev.STRUCT)
            areas = np.bincount(lab.ravel(), minlength=n + 1)[1:]
            mean_int = ndimage.mean(img[v], lab, np.arange(1, n + 1))
            key = f"{rec.source}|{rec.patient_id}|{view}"
            views.append({"key": key, "pid": pid, "hosp": hosp, "area": int(areas.sum()), "n_lesions": int(n)})
            for k in range(n):
                lesions.append({"key": key, "pid": pid, "hosp": hosp, "label": k + 1, "area": int(areas[k]),
                                "rel_intensity": float(mean_int[k])})
            views[-1]["_lab"] = lab  # kept in memory for detection below

    # detection of each reference lesion and per-view DSC for every method ---------------------------------------
    detection = {}
    dsc_rows = {}
    for m in METHODS:
        pred_files = [EVAL_DIR / f"{m}_fold{k}_pred.npz" for k in range(5)]
        row_files = [EVAL_DIR / f"{m}_fold{k}.json" for k in range(5)]
        if not all(p.exists() for p in pred_files + row_files):
            print("skip", m)
            continue
        npzs = [np.load(p) for p in pred_files]
        dsc = {}
        for p in row_files:
            for r in json.loads(p.read_text())["rows"]:
                dsc[f"{r['source']}|{r['patient_id']}|{r['view']}"] = r["dice"]
        hit = {}
        for vrow in views:
            key = vrow["key"]
            npz = next(z for z in npzs if key in z.files)
            pred = unpack(npz, key)
            hit_labels = set(np.unique(vrow["_lab"][pred])) - {0}
            for les in (x for x in lesions if x["key"] == key):
                hit[(key, les["label"])] = les["label"] in hit_labels
        detection[m] = hit
        dsc_rows[m] = [{"pid": v["pid"], "hosp": v["hosp"], "area": v["area"], "dsc": dsc[v["key"]]} for v in views]

    # summaries ------------------------------------------------------------------------------------------------
    out = {"n_boot": N_BOOT, "size_bins": SIZE_LABELS, "hospitals": {}, "tests": {}, "methods": {}}
    per_patient = defaultdict(lambda: defaultdict(list))
    for les in lesions:
        per_patient[les["hosp"]][les["pid"]].append(les)
    for h in ("A", "B"):
        hl = [x for x in lesions if x["hosp"] == h]
        hv = [x for x in views if x["hosp"] == h]
        areas = np.array([x["area"] for x in hl])
        out["hospitals"][h] = {
            "n_patients": len(per_patient[h]), "n_views": len(hv), "n_lesions": len(hl),
            "lesion_area": summary(areas),
            "share_lt10": float((areas < 10).mean()), "share_lt32": float((areas < 32).mean()),
            "share_lt100": float((areas < 100).mean()),
            "view_total_area": summary([x["area"] for x in hv]),
            "lesions_per_view": summary([x["n_lesions"] for x in hv]),
            "rel_intensity": summary([x["rel_intensity"] for x in hl]),
            "patient_median_area": summary([np.median([x["area"] for x in v]) for v in per_patient[h].values()]),
            "patient_median_intensity": summary([np.median([x["rel_intensity"] for x in v])
                                                 for v in per_patient[h].values()]),
            "areas": areas.tolist(),
        }
    for name, field in (("patient_median_area", "area"), ("patient_median_intensity", "rel_intensity")):
        a = [np.median([x[field] for x in v]) for v in per_patient["A"].values()]
        b = [np.median([x[field] for x in v]) for v in per_patient["B"].values()]
        out["tests"][name] = {"p_mannwhitney": float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)}
    va = [x["area"] for x in views if x["hosp"] == "A"]
    vb = [x["area"] for x in views if x["hosp"] == "B"]
    out["tests"]["view_total_area"] = {"p_mannwhitney": float(stats.mannwhitneyu(va, vb, alternative="two-sided").pvalue)}

    edges = np.array(SIZE_BINS)
    for m, hit in detection.items():
        entry = {"lesion_sensitivity": {}, "dsc_by_view_area": {}, "ols": ols_hospital(dsc_rows[m])}
        for h in ("A", "B"):
            hl = [x for x in lesions if x["hosp"] == h]
            idx = np.digitize([x["area"] for x in hl], edges) - 1
            entry["lesion_sensitivity"][h] = {
                "overall": float(np.mean([hit[(x["key"], x["label"])] for x in hl])),
                "by_size": [{"bin": SIZE_LABELS[b], "n": int((idx == b).sum()),
                             "sensitivity": float(np.mean([hit[(x["key"], x["label"])]
                                                           for x, j in zip(hl, idx) if j == b]))
                             if (idx == b).any() else None} for b in range(len(SIZE_LABELS))]}
        q = np.percentile([r["area"] for r in dsc_rows[m]], [25, 50, 75])
        for h in ("A", "B"):
            rr = [r for r in dsc_rows[m] if r["hosp"] == h]
            j = np.digitize([r["area"] for r in rr], q)
            entry["dsc_by_view_area"][h] = [{"quartile": b + 1, "n": int((j == b).sum()),
                                             "dsc_mean": float(np.mean([r["dsc"] for r, jj in zip(rr, j) if jj == b]))
                                             if (j == b).any() else None} for b in range(4)]
        entry["dsc_by_view_area"]["quartile_edges"] = [float(x) for x in q]
        entry["view_dsc"] = {h: [r["dsc"] for r in dsc_rows[m] if r["hosp"] == h] for h in ("A", "B")}
        entry["view_area"] = {h: [r["area"] for r in dsc_rows[m] if r["hosp"] == h] for h in ("A", "B")}
        out["methods"][m] = entry

    Path(sys.argv[1]).write_text(json.dumps(out))
    for h, s in out["hospitals"].items():
        print(f"{ {'A': 'CMUH', 'B': 'TTMHH'}[h]}: patients {s['n_patients']} views {s['n_views']} lesions {s['n_lesions']} "
              f"area median {s['lesion_area']['median']:.0f} (IQR {s['lesion_area']['q1']:.0f}-{s['lesion_area']['q3']:.0f}) "
              f"<32px {s['share_lt32']:.2f} view total median {s['view_total_area']['median']:.0f} "
              f"lesions/view median {s['lesions_per_view']['median']:.0f} mean {s['lesions_per_view']['mean']:.1f} "
              f"rel int median {s['rel_intensity']['median']:.3f}")
    print("tests", out["tests"])
    for m, e in out["methods"].items():
        o = e["ols"]
        print(f"{m:12s} sens A {e['lesion_sensitivity']['A']['overall']:.3f} B {e['lesion_sensitivity']['B']['overall']:.3f} "
              f"| gap raw {o['unadjusted_gap']:+.3f} adjusted {o['hospital_b_coef']:+.3f} "
              f"({o['hospital_b_ci'][0]:+.3f} to {o['hospital_b_ci'][1]:+.3f}) per-decade {o['log10_area_coef']:+.3f}")
        for h in ("A", "B"):
            print(f"   {h} sens by size", [(b["bin"], b["n"], None if b["sensitivity"] is None else round(b["sensitivity"], 3))
                                         for b in e["lesion_sensitivity"][h]["by_size"]])
            print(f"   {h} DSC by view-area quartile", [(b["quartile"], b["n"], None if b["dsc_mean"] is None else round(b["dsc_mean"], 3))
                                                    for b in e["dsc_by_view_area"][h]])


if __name__ == "__main__":
    main()
