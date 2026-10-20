"""Pick three lesion-positive patients (85th, 50th, 15th percentile of our per-patient Dice) and save the
anterior view, ground truth and every method's prediction for the qualitative figure.

Usage (from the repository root): python code/final_eval/extract_cases.py <eval_dir> <out_npz>
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BONESCAN = Path(os.environ.get("BONESCAN_ROOT", "."))  # folder with data/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the code/ folder
import bonescan.data as _bs_data  # noqa: E402

_bs_data.DEFAULT_OUT = BONESCAN / "data" / "standardized"
from bonescan.data import BoneScanDataset, assemble_patient_records, load_manifest  # noqa: E402

# first entry = reference for picking the cases: our final model (two-network ensemble)
METHODS = ["ours_ens", "ours_anat", "ours_noanat", "bsciseg", "nnunet"]
eval_dir, out_npz = Path(sys.argv[1]), Path(sys.argv[2])
ref = METHODS[0] if (eval_dir / f"{METHODS[0]}_fold0.json").exists() else "bsciseg"


def patient_dice(method):
    acc = defaultdict(lambda: [0, 0, 0])
    fold_of = {}
    for f in range(5):
        p = eval_dir / f"{method}_fold{f}.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["rows"]:
            if r["group"] == "cv" and r["has_lesion_mask"]:
                k = (r["source"], r["patient_id"])
                acc[k][0] += r["tp"]; acc[k][1] += r["fp"]; acc[k][2] += r["fn"]
                fold_of[k] = r["fold"]
    return {k: 2 * tp / (2 * tp + fp + fn) for k, (tp, fp, fn) in acc.items() if tp + fn > 0}, fold_of


ref_dice, fold_of = patient_dice(ref)
keys = sorted(ref_dice, key=ref_dice.get)
picks = []
for q in (0.85, 0.50, 0.15):
    target = np.quantile([ref_dice[k] for k in keys], q)
    k = min(keys, key=lambda kk: abs(ref_dice[kk] - target))
    if k not in picks:
        picks.append(k)

recs = {(r.source, r.patient_id): r
        for r in assemble_patient_records(load_manifest(_bs_data.DEFAULT_OUT), _bs_data.DEFAULT_OUT)}
ds = BoneScanDataset([recs[k] for k in picks], augment=False)
out = {}
for i, k in enumerate(picks):
    item = ds[i]
    out[f"c{i}_image"] = item["image"][0].numpy().astype(np.float32)
    out[f"c{i}_gt"] = (item["lesion_mask"][0].numpy() > 0).astype(np.uint8)
    out[f"c{i}_meta"] = np.array(json.dumps({"source": k[0], "patient_id": k[1], "fold": fold_of[k],
                                             "hospital": "B" if k[0].startswith("new") else "A"}))
    for m in METHODS:
        npz = eval_dir / f"{m}_fold{fold_of[k]}_pred.npz"
        if not npz.exists():
            continue
        z = np.load(npz)
        key = f"{k[0]}|{k[1]}|ANT"
        shape = tuple(z[key + "|shape"])
        out[f"c{i}_{m}"] = np.unpackbits(z[key])[: shape[0] * shape[1]].reshape(shape).astype(np.uint8)
    dices = {m: patient_dice(m)[0].get(k) for m in METHODS}
    out[f"c{i}_dice"] = np.array(json.dumps(dices))
np.savez_compressed(out_npz, **out)
print("picked", picks, "ref", ref)
