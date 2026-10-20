#!/usr/bin/env bash
# Aggregate the two-stage model against our one-stage model and the baselines (primary results, 0.5)
# with the same code as the primary analysis. Holm over three comparisons; the variant without anatomy features
# is descriptive only.
cd "${BONESCAN_ROOT:-.}"
# Uses the python on PATH: activate the environment of requirements.txt first.
D=runs/final_eval_2stage_cmp
rm -rf $D && mkdir -p $D
for k in 0 1 2 3 4; do
  cp runs/final_eval_2stage/full/ours_anat_fold$k.json $D/ours_anat_fold$k.json
  cp runs/final_eval_2stage/no_anatomy_features/ours_anat_fold$k.json $D/ours_2s_noanatfeat_fold$k.json
  cp runs/final_eval/ours_anat_fold$k.json $D/ours_1stage_fold$k.json
  cp runs/final_eval/bsciseg_fold$k.json runs/final_eval/nnunet_fold$k.json $D/
done
AGG_METHODS="ours_anat:Ours two-stage,ours_1stage:Ours one-stage,bsciseg:BSci-Seg (re-implemented),nnunet:nnU-Net (2D),ours_2s_noanatfeat:Ours two-stage without anatomy features" \
AGG_NOT_TESTED=ours_2s_noanatfeat python code/final_eval/aggregate_results.py $D $D/agg 2>&1 | grep -vE "^\s*$"
