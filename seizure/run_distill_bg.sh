#!/bin/bash
# Distillation variant with 30x background windows (see distill.py, TAG KDbg30).
cd "$HOME/bci-project" || exit 1
PY="./venv/bin/python"; LOG="$HOME/bci-project/seizure/progress.log"
export KD_BG_RATIO=30
for fold in 0 1 2 3 4 5 6; do
  echo "--- [KDbg30 fold$fold] $(date '+%T') ---" | tee -a "$LOG"
  $PY seizure/distill.py run "$fold" 2>>"$LOG" | tee -a "$LOG"
  sleep 3
done
echo "===== KDBG_DONE $(date '+%F %T') =====" | tee -a "$LOG"
