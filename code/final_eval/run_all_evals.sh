#!/usr/bin/env bash
# Evaluate every finished fold that has no up-to-date result yet, then aggregate and extract cases.
cd "${BONESCAN_ROOT:-.}"
# Uses the python on PATH: activate the environment of requirements.txt first.
GPU=${GPU:-1}
OUT=runs/final_eval
mkdir -p $OUT
NN=nnunet/results/Dataset501_BoneScanLesion/nnUNetTrainer_250epochs__nnUNetPlans__2d

ckpt_of() {  # method fold -> checkpoint path that marks the fold as finished
  case $1 in
    ours_anat)   echo runs/final_ours_anat/fold_$2/history.json ;;
    ours_noanat) echo runs/final_ours_noanat/fold_$2/history.json ;;
    ours_ens)    echo runs/final_ours_anat/fold_$2/history.json ;;
    ours_anat_kf) echo runs/final_ours_anat/fold_$2/history.json ;;
    bsciseg)     echo runs/ablation_BSciSeg/fold_$2/history.json ;;
    nnunet)      echo $NN/fold_$2/checkpoint_final.pth ;;
  esac
}

for m in ours_anat ours_noanat bsciseg nnunet ours_ens ours_anat_kf; do
  for f in 0 1 2 3 4; do
    done_marker=$(ckpt_of $m $f)
    res=$OUT/${m}_fold$f.json
    if [ -e "$done_marker" ] && { [ ! -e "$res" ] || [ "$done_marker" -nt "$res" ]; }; then
      CUDA_VISIBLE_DEVICES=$GPU METHOD=$m FOLDS=$f OUT_DIR=$OUT python -u code/final_eval/eval_final.py 2>&1 | grep -E "^\[$m"
    fi
  done
done
python code/final_eval/aggregate_results.py $OUT $OUT/agg 2>&1 | grep -vE "^\s*$"
python code/final_eval/extract_cases.py $OUT $OUT/agg/cases.npz 2>&1 | tail -1
