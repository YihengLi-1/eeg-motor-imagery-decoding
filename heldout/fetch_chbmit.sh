#!/bin/bash
# Parallel ranged download of BIDS CHB-MIT from Zenodo (per-connection speed is capped), resumable.
URL="https://zenodo.org/records/10259996/files/BIDS_CHB-MIT.zip?download=1"
SIZE=21686911024; N=48; D="$HOME/mne_data/chbmit_dl"
mkdir -p "$D/parts"; cd "$D/parts" || exit 1
CH=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  s=$((i*CH)); e=$(( s+CH-1 )); [ $e -ge $SIZE ] && e=$((SIZE-1))
  ( for try in $(seq 1 200); do
      have=$( [ -f p$i ] && stat -f%z p$i || echo 0 ); need=$((e-s+1))
      [ "$have" -ge "$need" ] && break
      curl -f -s -S -L --retry 5 --speed-time 120 --speed-limit 1000 -r $((s+have))-$e "$URL" >> p$i
    done ) &
done
wait
cat $(for i in $(seq 0 $((N-1))); do echo p$i; done) > "$D/BIDS_CHB-MIT.zip"
echo "md5 $(md5 -q "$D/BIDS_CHB-MIT.zip") expected df22b26fb8bf3db837c1bdbf205e99b7"
echo CHBMIT_FETCH_DONE
