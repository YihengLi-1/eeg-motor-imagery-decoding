#!/bin/zsh
# Cho2017 TimeMIL rerun under the MPS grouped-conv fix (run_queue_1101b.sh forgot MI_DATASET and ran PhysioNet instead).
cd ~/bci-project
export MI_DATASET=cho2017
for seed in 0 1 2 3; do
  venv/bin/python -W ignore heldout/timemil_heldout.py run broad_0.1-75 $seed
done
venv/bin/python -W ignore heldout/timemil_heldout.py summary
echo "[$(date '+%F %T')] TIMEMIL_CHO_DONE"
