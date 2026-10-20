#!/usr/bin/env bash
# Aggregate the nested two-stage pipeline against our one-stage model and the baselines (primary results)
# with the same code as the primary analysis (Holm over three comparisons); nnU-Net with its cross-validated
# threshold is descriptive only. Also summarises every fixed configuration (transparent reporting, not selection).
cd "${BONESCAN_ROOT:-.}"
# Uses the python on PATH: activate the environment of requirements.txt first.
N=runs/final_eval_nested
D=runs/final_eval_nested_cmp
rm -rf $D && mkdir -p $D
for k in 0 1 2 3 4; do
  cp $N/ours_anat_fold$k.json $D/ours_anat_fold$k.json
  cp runs/final_eval/ours_anat_fold$k.json $D/ours_1stage_fold$k.json
  cp runs/final_eval/bsciseg_fold$k.json runs/final_eval/nnunet_fold$k.json $D/
  cp runs/final_eval_cvthr/nnunet_fold$k.json $D/nnunet_cvthr_fold$k.json
done
AGG_METHODS="ours_anat:Ours nested two-stage,ours_1stage:Ours one-stage,bsciseg:BSci-Seg (re-implemented),nnunet:nnU-Net (2D),nnunet_cvthr:nnU-Net with cross-validated threshold" \
AGG_NOT_TESTED=nnunet_cvthr python code/final_eval/aggregate_results.py $D $D/agg 2>&1 | grep -vE "^\s*$"
python - <<'PY'
import json, glob, os
import numpy as np
out = {}
for d in sorted(glob.glob("runs/final_eval_nested/fixed/*")):
    rows = [r for f in sorted(glob.glob(d + "/ours_anat_fold*.json")) for r in json.load(open(f))["rows"]]
    pos = [r for r in rows if r["group"] == "cv" and r["has_lesion_mask"] and r["gt_area"] > 0]
    out[os.path.basename(d)] = {"dsc_img": float(np.mean([r["dice"] for r in pos])),
                                "lesion_sensitivity": sum(r["n_gt_detected"] for r in pos) / sum(r["n_gt_lesions"] for r in pos)}
json.dump(out, open("runs/final_eval_nested/fixed_summary.json", "w"), indent=2)
for k, v in sorted(out.items(), key=lambda kv: -kv[1]["dsc_img"]):
    print(f"  fixed {k:22s} DSC_img {v['dsc_img']:.4f}  lesion sensitivity {v['lesion_sensitivity']:.3f}")
PY
