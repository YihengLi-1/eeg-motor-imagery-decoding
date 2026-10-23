"""Load every Cho2017 .mat with zlib integrity checks; print the subjects that fail."""
import os, sys
from scipy.io import loadmat
bad = []
for i in range(1, 53):
    f = os.path.expanduser(f"~/mne_data/cho2017/s{i:02d}.mat")
    try:
        loadmat(f, squeeze_me=True, struct_as_record=False)["eeg"].imagery_left.shape
    except Exception as e:
        bad.append(i); print(f"s{i:02d} FAIL {e}", flush=True)
print("CHO_VERIFY", "OK" if not bad else f"BAD {bad}")
sys.exit(1 if bad else 0)
