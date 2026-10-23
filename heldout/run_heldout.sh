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

# ---- 1. CHB-MIT download -> md5 -> unzip ----
say "waiting for CHB-MIT download"
until grep -q CHBMIT_FETCH_DONE heldout/fetch_chbmit.log 2>/dev/null; do sleep 120; done
md5line=$(grep "^md5" heldout/fetch_chbmit.log)
[ "$(echo $md5line | awk '{print $2}')" = "df22b26fb8bf3db837c1bdbf205e99b7" ] || fail "CHB-MIT md5 mismatch: $md5line"
if [ ! -d "$HOME/mne_data/BIDS_CHB-MIT" ]; then
  say "unzipping CHB-MIT"
  mkdir -p "$HOME/mne_data/chbmit_unzip" && unzip -q -o "$HOME/mne_data/chbmit_dl/BIDS_CHB-MIT.zip" -d "$HOME/mne_data/chbmit_unzip" || fail "unzip"
  top=$(dirname "$(find "$HOME/mne_data/chbmit_unzip" -maxdepth 3 -type d -name 'sub-01' | head -1)")
  [ -n "$top" ] || fail "no sub-01 in CHB-MIT zip"
  ln -sfn "$top" "$HOME/mne_data/BIDS_CHB-MIT"
fi
say "CHB-MIT subjects: $(ls -d $HOME/mne_data/BIDS_CHB-MIT/sub-* | wc -l)"
SZ_DATASET=chbmit $PY seizure/szcore_siena.py prep 2>>"$LOG" | tee -a "$LOG" || fail "CHB-MIT prep"

# ---- 2. Cho2017 download -> size check -> prep ----
say "waiting for Cho2017 download"
until grep -q CHO_FETCH_DONE heldout/fetch_cho2017.log 2>/dev/null; do sleep 120; done
BASE="https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100295/mat_data"
for i in $(seq 1 52); do
  f=$(printf "s%02d.mat" $i)
  for try in 1 2 3; do
    want=$(curl -sIL "$BASE/$f" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r' | tail -1)
    have=$(stat -f%z "$HOME/mne_data/cho2017/$f" 2>/dev/null || echo 0)
    [ "$have" = "$want" ] && break
    say "re-fetching $f ($have/$want)"; curl -s -L -C - -o "$HOME/mne_data/cho2017/$f" "$BASE/$f"
  done
  [ "$have" = "$want" ] || fail "Cho2017 $f incomplete"
done
say "Cho2017: all 52 files complete"
MI_DATASET=cho2017 $PY step3d_matrix.py prep 2>>"$LOG" | tee -a "$LOG" || fail "Cho2017 prep"

# ---- 3. wait for the GPU (Siena LOSO) ----
say "waiting for Siena LOSO to finish"
while pgrep -f run_loso.sh >/dev/null; do sleep 300; done

# ---- 4. smoke tests (not results) ----
say "smoke tests"
for kind in EEGNet fullFT; do
  SMOKE=1 FORCE=1 SZ_DATASET=chbmit $PY seizure/szcore_siena.py run $kind 0 2>>"$LOG" | tee -a "$LOG" || fail "CHB-MIT smoke $kind"
done
rm -f seizure/results_szcore_chbmit.json seizure/_models_chbmit/*      # smoke output is not a result

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
