#!/bin/bash
# Leave-one-patient-out version of the seizure study (SzCORE's recommended subject-independent CV).
cd "$HOME/bci-project" || exit 1
PY="./venv/bin/python"; LOG="$HOME/bci-project/seizure/progress_loso.log"
export SZ_SCHEME=loso TOKENIZERS_PARALLELISM=false
echo "===== loso start $(date '+%F %T') =====" | tee -a "$LOG"
for kind in fullFT EEGNet; do
  for fold in $(seq 0 13); do
    echo "--- [$kind loso fold$fold] $(date '+%T') ---" | tee -a "$LOG"
    $PY seizure/szcore_siena.py run "$kind" "$fold" 2>>"$LOG" | tee -a "$LOG"
    sleep 3
  done
done
$PY seizure/szcore_siena.py summary 2>>"$LOG" | tee -a "$LOG"
$PY seizure/repostproc.py EEGNet fullFT 2>>"$LOG" | tee -a "$LOG"
$PY seizure/ensemble.py 2>>"$LOG" | tee -a "$LOG"
echo "===== LOSO_DONE $(date '+%F %T') =====" | tee -a "$LOG"
