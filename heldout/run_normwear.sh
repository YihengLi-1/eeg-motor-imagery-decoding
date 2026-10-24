#!/bin/bash
# NormWear on Cho2017: both preprocessing conditions, then the probe. Safe to re-run (embed overwrites).
set -eo pipefail
cd ~/bci-project/heldout
for c in paper nosmooth; do
  [ -f _normwear_emb_$c.npz ] || ../venv/bin/python -W ignore normwear_cho2017.py embed $c
done
../venv/bin/python -W ignore normwear_cho2017.py probe
echo NORMWEAR_DONE
