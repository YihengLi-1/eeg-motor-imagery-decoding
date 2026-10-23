#!/bin/bash
# Cho et al. 2017 (GigaDB 100295) left/right-hand motor imagery, 52 subjects, one .mat per subject (~200 MB each).
BASE="https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100295/mat_data"
D="$HOME/mne_data/cho2017"; mkdir -p "$D"; cd "$D" || exit 1
fetch() {
  f=$(printf "s%02d.mat" "$1")
  for try in $(seq 1 10); do
    curl -s -S -L --retry 5 -C - -o "$f" "$BASE/$f" && break
  done
}
export -f fetch; export BASE
seq 1 52 | xargs -P 24 -I{} bash -c 'fetch {}'
ls s*.mat | wc -l
echo CHO_FETCH_DONE
