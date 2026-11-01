#!/bin/bash
# 10/31 night queue: (1) random-init controls on Cho2017 (correct ablation for the pretrained weights), (2) live bench season 1 on ds008446.
cd "$HOME/bci-project" || exit 1
LOG="$HOME/bci-project/heldout/queue_1031.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
export TOKENIZERS_PARALLELISM=false
say "START queue"
for kind in randFT cbramodRand; do
  for seed in 0 1 2 3; do
    say "cho2017 $kind seed $seed"
    MI_DATASET=cho2017 venv/bin/python -W ignore step3d_matrix.py run $kind broad_0.1-75 $seed 2>&1 | grep -v -i warn | tail -3 | tee -a "$LOG"
  done
done
MI_DATASET=cho2017 venv/bin/python -W ignore step3d_matrix.py summary 2>&1 | tail -14 | tee -a "$LOG"
say "RANDINIT_DONE"
say "waiting for ds008446 download"
until grep -q "FETCH_DONE ds008446" heldout/fetch_ds008446_s3.log 2>/dev/null; do sleep 60; done
say "live prep ds008446"
LIVE_DATASET=ds008446 venv/bin/python -W ignore heldout/live_heldout.py prep 2>&1 | grep -v -i "warn\|deprecat" | tail -30 | tee -a "$LOG"
say "live smoke EEGNet"
SMOKE=1 LIVE_DATASET=ds008446 venv/bin/python -W ignore heldout/live_heldout.py run EEGNet 0 2>&1 | grep -v -i warn | tail -4 | tee -a "$LOG"
for kind in EEGNet probe fullFT cbramodProbe cbramodFT; do
  for seed in 0 1 2 3; do
    say "live $kind seed $seed"
    LIVE_DATASET=ds008446 venv/bin/python -W ignore heldout/live_heldout.py run $kind $seed 2>&1 | grep -v -i warn | tail -3 | tee -a "$LOG"
  done
done
LIVE_DATASET=ds008446 venv/bin/python -W ignore heldout/live_heldout.py summary 2>&1 | tail -12 | tee -a "$LOG"
say "QUEUE_DONE"
