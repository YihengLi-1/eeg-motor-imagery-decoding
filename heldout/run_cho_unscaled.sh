#!/bin/bash
# Sensitivity analysis after the main queue: LaBraM fine-tuning on Cho2017 at its raw (MOABB, CAR) amplitude.
cd "$HOME/bci-project" || exit 1
PY="./venv/bin/python"; LOG="$HOME/bci-project/heldout/progress.log"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 MI_DATASET=cho2017 CHO_SCALE=1
until grep -q -E "HELDOUT_DONE|HELDOUT_FAILED" "$LOG"; do sleep 600; done
grep -q HELDOUT_FAILED "$LOG" && exit 1
for seed in 0 1 2 3; do
  echo "[$(date '+%F %T')] [cho2017 unscaled fullFT/broad seed$seed]" | tee -a "$LOG"
  $PY step3d_matrix.py run fullFT broad_0.1-75 "$seed" 2>>"$LOG" | tee -a "$LOG"; sleep 3
done
$PY step3d_matrix.py summary 2>>"$LOG" | tee -a "$LOG"
echo "[$(date '+%F %T')] UNSCALED_DONE" | tee -a "$LOG"
