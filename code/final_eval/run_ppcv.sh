#!/usr/bin/env bash
# Secondary analysis with post-processing chosen by cross-validation.
# 1) score every post-processing configuration for each finished method-fold without an up-to-date *_pp.npz,
# 2) choose each fold's configuration on the other four folds (postproc_cv.py), 3) aggregate with the same code.
cd "${BONESCAN_ROOT:-.}"
# Uses the python on PATH: activate the environment of requirements.txt first.
GPU=${GPU:-0}
PP=runs/final_eval_pp
CV=runs/final_eval_ppcv
NN=nnunet/results/Dataset501_BoneScanLesion/nnUNetTrainer_250epochs__nnUNetPlans__2d
mkdir -p $PP $CV

ckpt_of() {
  case $1 in
    ours_anat|ours_ens|ours_anat_kf) echo runs/final_ours_anat/fold_$2/history.json ;;
    ours_noanat) echo runs/final_ours_noanat/fold_$2/history.json ;;
    bsciseg)     echo runs/ablation_BSciSeg/fold_$2/history.json ;;
    nnunet)      echo $NN/fold_$2/checkpoint_final.pth ;;
  esac
}

for m in ${METHODS:-ours_anat ours_noanat bsciseg nnunet ours_ens ours_anat_kf}; do
  for f in 0 1 2 3 4; do
    marker=$(ckpt_of $m $f)
    res=$PP/${m}_fold${f}_pp.npz
    if [ -e "$marker" ] && { [ ! -e "$res" ] || [ "$marker" -nt "$res" ]; }; then
      CUDA_VISIBLE_DEVICES=$GPU METHOD=$m FOLDS=$f PP_GRID=1 OUT_DIR=$PP python -u code/final_eval/eval_final.py 2>&1 | grep -E "^\[$m|Error|Traceback"
    fi
  done
done
[ -n "$SKIP_SELECT" ] && exit 0
python code/final_eval/postproc_cv.py $PP $CV runs/final_eval
python code/final_eval/aggregate_results.py $CV $CV/agg 2>&1 | grep -vE "^\s*$"
