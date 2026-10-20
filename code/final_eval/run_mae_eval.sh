#!/usr/bin/env bash
# Evaluate the network trained without MAE pre-training (final_ours_anat_nomae, all five folds present) with
# the same code, then compare it with the network trained with MAE (paired, unadjusted, same aggregation code).
cd "${BONESCAN_ROOT:-.}"
# Uses the python on PATH: activate the environment of requirements.txt first.
for f in 0 1 2 3 4; do [ -f runs/final_ours_anat_nomae/fold_$f/best.pt ] || { echo "missing fold $f"; exit 1; }; done
CUDA_VISIBLE_DEVICES=${GPU:-0} METHOD=ours_anat RUN_NAME=final_ours_anat_nomae OUT_DIR=runs/final_eval_nomae FOLDS=0,1,2,3,4 \
  python -u code/final_eval/eval_final.py 2>&1 | grep -E "^\[ours|Error|Traceback"
D=runs/final_mae_cmp
rm -rf $D && mkdir -p $D
for k in 0 1 2 3 4; do
  cp runs/final_eval/ours_anat_fold$k.json $D/ours_anat_fold$k.json
  cp runs/final_eval_nomae/ours_anat_fold$k.json $D/ours_nomae_fold$k.json
done
AGG_METHODS="ours_anat:Single network with anatomy task (MAE),ours_nomae:Single network with anatomy task (no MAE)" \
  python code/final_eval/aggregate_results.py $D $D/agg 2>&1 | grep -vE "^\s*$"
