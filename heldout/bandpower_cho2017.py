"""Sanity baseline for the NormWear result: log band power of the same 9 motor channels + the same logistic-regression
probe and the same 4 subject splits as normwear_cho2017.py / step3d_matrix.py."""
import os, json, pickle
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = os.path.expanduser("~/bci-project")
CHANS = ["FC3", "FC4", "C3", "C1", "Cz", "C2", "C4", "CP3", "CP4"]
BANDS = [(4, 8), (8, 10), (10, 13), (13, 20), (20, 30), (30, 40)]
meta = pickle.load(open(f"{ROOT}/heldout/_cache_cho2017_meta.pkl", "rb")); clean = meta["clean"]
idx = [[c["ch_name"] for c in meta["chs_info"]].index(c) for c in CHANS]
D = pickle.load(open(f"{ROOT}/heldout/_cache_cho2017_broad_0.1-75.pkl", "rb"))
def feats(X):
    f, P = welch(X[:, idx, 200:], fs=200, nperseg=200, axis=-1)          # 0..4 s after cue
    return np.log(np.stack([P[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in BANDS], -1)).reshape(len(X), -1)
E = {s: (feats(D[s][0]), D[s][1]) for s in D}
accs = []
for seed in [0, 1, 2, 3]:
    perm = np.random.RandomState(seed).permutation(clean); te, va, tr = list(perm[:10]), list(perm[10:15]), list(perm[15:])
    g = lambda ss: (np.concatenate([E[s][0] for s in ss]), np.concatenate([E[s][1] for s in ss]))
    (Xtr, ytr), (Xva, yva) = g(tr), g(va)
    best = max((make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(Xtr, ytr)
                for C in (1e-4, 1e-3, 1e-2, 1e-1)), key=lambda m: m.score(Xva, yva))
    accs.append(float(np.mean([best.score(*g([s])) for s in te])))
print(f"log band power (9 ch x {len(BANDS)} bands) + LR: {np.mean(accs)*100:.1f} +- {np.std(accs)*100:.1f}  per seed {np.round(np.array(accs)*100,1)}")
json.dump({"bandpower_lr": {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)), "per_seed": accs}},
          open(f"{ROOT}/heldout/results_cho2017_bandpower.json", "w"), indent=1)
