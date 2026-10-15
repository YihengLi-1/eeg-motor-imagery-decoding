#!/bin/bash
# Parallel ranged download of BIDS Siena from Zenodo (per-connection speed is capped ~40 KB/s).
URL="https://zenodo.org/records/10640762/files/BIDS_Siena.zip?download=1"
SIZE=4489331248; N=32; D="$HOME/mne_data/siena_dl"
cd "$D/parts" || exit 1
CH=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  s=$((i*CH)); e=$(( s+CH-1 )); [ $e -ge $SIZE ] && e=$((SIZE-1))
  ( for try in 1 2 3 4 5 6 7 8; do
      have=$( [ -f p$i ] && stat -f%z p$i || echo 0 ); need=$((e-s+1))
      [ "$have" -ge "$need" ] && break
      curl -s -S -L --retry 5 -r $((s+have))-$e "$URL" >> p$i
    done ) &
done
wait
cat $(for i in $(seq 0 $((N-1))); do echo p$i; done) > "$D/BIDS_Siena.zip"
echo "md5 $(md5 -q "$D/BIDS_Siena.zip") expected 3145f577a6caa35a84f89f4c6ce0f9d7"
