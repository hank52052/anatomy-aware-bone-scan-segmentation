#!/usr/bin/env bash
# Secondary analysis: score every threshold of the grid for each method and fold (writes runs/final_eval_thr).
# Usage: GPU=0 bash run_thr_grid.sh "method:folds method:folds ..."
cd "${BONESCAN_ROOT:-.}"
# Uses the python on PATH: activate the environment of requirements.txt first.
OUT=runs/final_eval_thr
mkdir -p $OUT
for job in $1; do
  m=${job%%:*}; folds=${job#*:}
  CUDA_VISIBLE_DEVICES=${GPU:-0} METHOD=$m FOLDS=$folds THRESH_GRID=1 OUT_DIR=$OUT python -u code/final_eval/eval_final.py 2>&1 | grep -E "^\[$m|Error"
done
echo DONE
