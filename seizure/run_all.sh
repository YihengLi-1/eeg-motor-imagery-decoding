#!/bin/bash
# Same pattern as run_matrix_robust.sh: one process per (model, fold), resumable, kept awake.
cd "$HOME/bci-project" || exit 1
PY="./venv/bin/python"
LOG="$HOME/bci-project/seizure/progress.log"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TOKENIZERS_PARALLELISM=false

echo "===== start $(date '+%F %T') =====" | tee -a "$LOG"
$PY seizure/szcore_siena.py prep 2>>"$LOG" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "PREP FAILED, stopping" | tee -a "$LOG"; exit 1; }
# LaBraM first: it is the result the application needs most
for kind in fullFT EEGNet; do
  for fold in 0 1 2 3 4 5 6; do
    echo "--- [$kind fold$fold] $(date '+%T') ---" | tee -a "$LOG"
    $PY seizure/szcore_siena.py run "$kind" "$fold" 2>>"$LOG" | tee -a "$LOG"
    sleep 3
  done
done
$PY seizure/szcore_siena.py summary 2>>"$LOG" | tee -a "$LOG"
echo "===== ALL_DONE $(date '+%F %T') =====" | tee -a "$LOG"
