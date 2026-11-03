#!/bin/bash
# Follow-up queue after run_queue_1101.sh: TimeMIL rerun on Cho2017 (grouped conv1d, batch 32 -> MPS bug), 4 seeds.
# Seizure (Siena/CHB-MIT) and edge_quant reruns are NOT started automatically: they take hours and need a decision.
cd "$HOME/bci-project" || exit 1
LOG="$HOME/bci-project/heldout/queue_1101.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
until grep -q "QUEUE_DONE" "$LOG" 2>/dev/null; do sleep 120; done
say "START queue b: TimeMIL rerun"
cp heldout/results_cho2017_timemil.json heldout/_backup_20261101_0011/results_cho2017_timemil.json 2>/dev/null
rm -f heldout/results_cho2017_timemil.json
for seed in 0 1 2 3; do
  say "cho2017 TimeMIL seed $seed"
  venv/bin/python -W ignore heldout/timemil_heldout.py run broad_0.1-75 $seed 2>&1 | grep -v -i warn | tail -3 | tee -a "$LOG"
done
venv/bin/python -W ignore heldout/timemil_heldout.py summary 2>&1 | tail -8 | tee -a "$LOG"
say "TIMEMIL_DONE"
