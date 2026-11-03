#!/bin/zsh
# 2026-11-01: rerun every EEGNet-based seizure result (and EEGNet edge quantisation) after the MPS grouped-conv fix
# (docs/mps_groupconv_bug.md). fullFT (LaBraM) rows are untouched. Old EEGNet rows/weights were backed up to
# seizure/_backup_20261101_prefix_mps and removed, so each script's own "done, skip" logic reruns exactly those.
cd ~/bci-project || exit 1
PY=./venv/bin/python; LOG=~/bci-project/seizure/progress_rerun_1101.log
export TOKENIZERS_PARALLELISM=false
echo "===== rerun start $(date '+%F %T') =====" | tee -a $LOG
# 1. Siena, 7 folds
for fold in 0 1 2 3 4 5 6; do
  echo "--- [EEGNet fold$fold] $(date '+%T') ---" | tee -a $LOG
  $PY -W ignore seizure/szcore_siena.py run EEGNet $fold 2>>$LOG | tee -a $LOG
done
$PY -W ignore seizure/szcore_siena.py summary 2>>$LOG | tee -a $LOG
echo "SIENA_K7_EEGNET_DONE $(date '+%T')" | tee -a $LOG
for fold in 0 1 2 3 4 5 6; do
  echo "--- [KD fold$fold] $(date '+%T') ---" | tee -a $LOG
  $PY -W ignore seizure/distill.py run $fold 2>>$LOG | tee -a $LOG
done
for fold in 0 1 2 3 4 5 6; do
  echo "--- [KDbg30 fold$fold] $(date '+%T') ---" | tee -a $LOG
  KD_BG_RATIO=30 $PY -W ignore seizure/distill.py run $fold 2>>$LOG | tee -a $LOG
done
$PY -W ignore seizure/distill.py summary 2>>$LOG | tee -a $LOG
$PY -W ignore seizure/repostproc.py EEGNet fullFT KD KDbg30 2>>$LOG | tee -a $LOG
$PY -W ignore seizure/ensemble.py 2>>$LOG | tee -a $LOG
echo "SIENA_K7_DONE $(date '+%T')" | tee -a $LOG
# 2. Siena, leave-one-patient-out
for fold in $(seq 0 13); do
  echo "--- [EEGNet loso fold$fold] $(date '+%T') ---" | tee -a $LOG
  SZ_SCHEME=loso $PY -W ignore seizure/szcore_siena.py run EEGNet $fold 2>>$LOG | tee -a $LOG
done
SZ_SCHEME=loso $PY -W ignore seizure/szcore_siena.py summary 2>>$LOG | tee -a $LOG
SZ_SCHEME=loso $PY -W ignore seizure/repostproc.py EEGNet fullFT 2>>$LOG | tee -a $LOG
SZ_SCHEME=loso $PY -W ignore seizure/ensemble.py 2>>$LOG | tee -a $LOG
echo "SIENA_LOSO_DONE $(date '+%T')" | tee -a $LOG
# 3. CHB-MIT, 8 folds
for fold in 0 1 2 3 4 5 6 7; do
  echo "--- [EEGNet chbmit fold$fold] $(date '+%T') ---" | tee -a $LOG
  SZ_DATASET=chbmit $PY -W ignore seizure/szcore_siena.py run EEGNet $fold 2>>$LOG | tee -a $LOG
done
SZ_DATASET=chbmit $PY -W ignore seizure/szcore_siena.py summary 2>>$LOG | tee -a $LOG
SZ_DATASET=chbmit $PY -W ignore seizure/repostproc.py EEGNet fullFT 2>>$LOG | tee -a $LOG
SZ_DATASET=chbmit $PY -W ignore seizure/ensemble.py 2>>$LOG | tee -a $LOG
echo "CHBMIT_DONE $(date '+%T')" | tee -a $LOG
# 4. edge quantisation, EEGNet seeds
for seed in 0 1 2 3; do
  echo "--- [edge EEGNet seed$seed] $(date '+%T') ---" | tee -a $LOG
  $PY -W ignore heldout/edge_quant.py EEGNet $seed 2>>$LOG | tee -a $LOG
done
echo "EDGE_DONE $(date '+%T')" | tee -a $LOG
# 5. derived tables/figures (Siena k7)
$PY -W ignore seizure/per_subject_stats.py 2>>$LOG | tee -a $LOG
echo "===== RERUN_ALL_DONE $(date '+%F %T') =====" | tee -a $LOG
