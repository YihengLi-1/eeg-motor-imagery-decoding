#!/bin/bash
# Runs after run_all.sh: distil every fold from its saved LaBraM checkpoint,
# then refresh both summaries.
cd "$HOME/bci-project" || exit 1
PY="./venv/bin/python"; LOG="$HOME/bci-project/seizure/progress.log"
until grep -q "ALL_DONE" "$LOG"; do sleep 60; done
echo "===== distill stage $(date '+%F %T') =====" | tee -a "$LOG"

$PY seizure/szcore_siena.py summary 2>>"$LOG" | tee -a "$LOG"
for fold in 0 1 2 3 4 5 6; do
  echo "--- [KD fold$fold] $(date '+%T') ---" | tee -a "$LOG"
  $PY seizure/distill.py run "$fold" 2>>"$LOG" | tee -a "$LOG"
  sleep 3
done
$PY seizure/distill.py summary 2>>"$LOG" | tee -a "$LOG"
echo "===== DISTILL_DONE $(date '+%F %T') =====" | tee -a "$LOG"
