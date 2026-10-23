#!/bin/bash
# Held-out replication queue: datasets outside LaBraM's pretraining corpus.
#   CHB-MIT (seizure, SzCORE)  and  Cho2017 (motor imagery), same protocols as the original studies.
# Order: verify + unzip + prep on CPU while the Siena LOSO run holds the GPU, then smoke tests, then full runs.
cd "$HOME/bci-project" || exit 1
PY="./venv/bin/python"; LOG="$HOME/bci-project/heldout/progress.log"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
set -o pipefail
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
fail() { say "STOP: $*"; say "HELDOUT_FAILED"; exit 1; }
say "resume after reboot"
# ---- 5. CHB-MIT full run ----
export SZ_DATASET=chbmit
for kind in fullFT EEGNet; do
  for fold in $(seq 0 7); do
    say "[chbmit $kind fold$fold]"
    $PY seizure/szcore_siena.py run "$kind" "$fold" 2>>"$LOG" | tee -a "$LOG"
    sleep 3
  done
done
$PY seizure/szcore_siena.py summary 2>>"$LOG" | tee -a "$LOG"
$PY seizure/repostproc.py EEGNet fullFT 2>>"$LOG" | tee -a "$LOG"
$PY seizure/ensemble.py 2>>"$LOG" | tee -a "$LOG"
say "CHBMIT_DONE"
unset SZ_DATASET

# ---- 6. Cho2017 motor-imagery matrix (6 configs x 4 splits) ----
export MI_DATASET=cho2017
for cfg in "fullFT broad_0.1-75" "EEGNet broad_0.1-75" "fullFT narrow_8-30" "EEGNet narrow_8-30" "probe broad_0.1-75" "probe narrow_8-30"; do
  set -- $cfg
  for seed in 0 1 2 3; do
    say "[cho2017 $1/$2 seed$seed]"
    $PY step3d_matrix.py run "$1" "$2" "$seed" 2>>"$LOG" | tee -a "$LOG"
    sleep 3
  done
done
$PY step3d_matrix.py summary 2>>"$LOG" | tee -a "$LOG"
say "HELDOUT_DONE"
