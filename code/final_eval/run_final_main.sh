#!/usr/bin/env bash
# Main comparison with our final model (two-network ensemble) as the reference, same aggregation code.
# Holm over three comparisons (BSci-Seg, nnU-Net, single network with the anatomy task); the single network
# without the anatomy task is descriptive here (its ablation comparison comes from the single-network summary).
cd "${BONESCAN_ROOT:-.}"
# Uses the python on PATH: activate the environment of requirements.txt first.
D=runs/final_main_cmp
rm -rf $D && mkdir -p $D
for k in 0 1 2 3 4; do
  cp runs/final_eval/ours_ens_fold$k.json $D/ours_anat_fold$k.json
  cp runs/final_eval/ours_anat_fold$k.json $D/ours_1stage_fold$k.json
  cp runs/final_eval/ours_noanat_fold$k.json runs/final_eval/bsciseg_fold$k.json runs/final_eval/nnunet_fold$k.json $D/
done
AGG_METHODS="ours_anat:Ours (two-network ensemble),ours_1stage:Single network with anatomy task,ours_noanat:Single network without anatomy task,bsciseg:BSci-Seg (re-implemented),nnunet:nnU-Net (2D)" \
AGG_NOT_TESTED=ours_noanat python code/final_eval/aggregate_results.py $D $D/agg 2>&1 | grep -vE "^\s*$"
