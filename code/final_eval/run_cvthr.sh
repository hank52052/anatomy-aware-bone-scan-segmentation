#!/usr/bin/env bash
# Secondary analysis with thresholds chosen by cross-validation.
# 1) score the threshold grid for every finished method-fold that has no up-to-date *_thr.json yet,
# 2) choose each fold's threshold on the other four folds (threshold_cv.py), 3) aggregate with the same code.
cd "${BONESCAN_ROOT:-.}"
# Uses the python on PATH: activate the environment of requirements.txt first.
GPU=${GPU:-0}
THR=runs/final_eval_thr
CV=runs/final_eval_cvthr
NN=nnunet/results/Dataset501_BoneScanLesion/nnUNetTrainer_250epochs__nnUNetPlans__2d
mkdir -p $THR $CV

ckpt_of() {
  case $1 in
    ours_anat|ours_ens|ours_anat_kf) echo runs/final_ours_anat/fold_$2/history.json ;;
    ours_noanat) echo runs/final_ours_noanat/fold_$2/history.json ;;
    bsciseg)     echo runs/ablation_BSciSeg/fold_$2/history.json ;;
    nnunet)      echo $NN/fold_$2/checkpoint_final.pth ;;
  esac
}

for m in ours_anat ours_noanat bsciseg nnunet ours_ens ours_anat_kf; do
  for f in 0 1 2 3 4; do
    marker=$(ckpt_of $m $f)
    res=$THR/${m}_fold${f}_thr.json
    if [ -e "$marker" ] && { [ ! -e "$res" ] || [ "$marker" -nt "$res" ]; }; then
      CUDA_VISIBLE_DEVICES=$GPU METHOD=$m FOLDS=$f THRESH_GRID=1 OUT_DIR=$THR python -u code/final_eval/eval_final.py 2>&1 | grep -E "^\[$m|Error"
    fi
  done
done
python code/final_eval/threshold_cv.py $THR $CV
python code/final_eval/aggregate_results.py $CV $CV/agg 2>&1 | grep -vE "^\s*$"
