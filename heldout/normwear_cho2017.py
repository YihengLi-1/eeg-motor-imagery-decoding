"""NormWear (Luo, Chen, Salekin, Rahman; ACM HEALTH 2026) on Cho2017 left/right motor imagery, cross-subject.

Same subject splits as step3d_matrix.py (seeds 0-3: 10 test / 5 val / rest train), so the numbers sit next to
EEGNet / LaBraM in results_cho2017_matrix.json. Frozen encoder + logistic regression, as in the NormWear paper.

Two preprocessing conditions, everything else fixed:
  paper    : NormWear Appendix B (65 Hz, linear detrend, Gaussian smoothing sigma=1.3 samples, /mean|x|)
  nosmooth : identical but without the Gaussian smoothing (sigma=0.02 s attenuates 10 Hz to 0.45x, 20 Hz to 0.04x)
Usage:  python normwear_cho2017.py embed <paper|nosmooth>   # cache embeddings
        python normwear_cho2017.py probe                    # logistic-regression probe, writes results json
"""
import os, sys, json, pickle, time
import numpy as np

ROOT = os.path.expanduser("~/bci-project")
NW_SRC = os.environ.get("NW_SRC", os.path.expanduser("~/mne_data/normwear_src"))   # parent of the NormWear clone
CKPT = os.path.expanduser("~/mne_data/normwear_ckpt/normwear_pretrain_ckpt.pth")
BAND = "broad_0.1-75"                        # cached epochs: 200 Hz, -1..4 s, uV, CAR
CHANS = ["FC3", "FC4", "C3", "C1", "Cz", "C2", "C4", "CP3", "CP4"]
SR_IN, SR_NW, WIN_S = 200, 65, 6             # NormWear: 65 Hz, 6 s windows, zero-padded
SEEDS = [0, 1, 2, 3]
EMB = lambda cond: f"{ROOT}/heldout/_normwear_emb_{cond}.npz"
OUT = f"{ROOT}/heldout/results_cho2017_normwear.json"


def preprocess(x, smooth):
    """x: [n, c, L] at 200 Hz -> [n, c, 390] at 65 Hz following NormWear Appendix B."""
    from scipy.signal import resample_poly, detrend
    from scipy.ndimage import gaussian_filter1d
    x = resample_poly(x, 13, 40, axis=-1)                  # 200 -> 65 Hz
    x = detrend(x, axis=-1)
    if smooth:
        x = gaussian_filter1d(x, sigma=SR_NW * 0.02, axis=-1)
    x = x / (np.abs(x).mean(-1, keepdims=True) + 1e-8)
    pad = SR_NW * WIN_S - x.shape[-1]
    return np.pad(x, ((0, 0), (0, 0), (0, pad))).astype("float32")


def self_check():
    """Smoothing must pass slow waves and kill beta; nosmooth must keep beta."""
    t = np.arange(1000) / SR_IN                            # 5 s -> 325 samples at 65 Hz, 0.2 Hz bins
    x = (np.sin(2 * np.pi * 3 * t) + np.sin(2 * np.pi * 20 * t))[None, None]
    beta_ratio = lambda y: (lambda F: F[100] / F[15])(np.abs(np.fft.rfft(y[0, 0, :325])))   # 20 Hz / 3 Hz
    r = beta_ratio(preprocess(x, True)) / beta_ratio(preprocess(x, False))
    assert r < 0.1, r                                      # theory: exp(-2 pi^2 0.02^2 (20^2 - 3^2)) = 0.046


def embed(cond):
    import torch
    sys.path.insert(0, NW_SRC)
    from NormWear.main_model import NormWearModel
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = NormWearModel(weight_path=CKPT, optimized_cwt=True).to(dev).eval()
    meta = pickle.load(open(f"{ROOT}/heldout/_cache_cho2017_meta.pkl", "rb"))
    names = [c["ch_name"] for c in meta["chs_info"]]
    idx = [names.index(c) for c in CHANS]
    D = pickle.load(open(f"{ROOT}/heldout/_cache_cho2017_{BAND}.pkl", "rb"))
    out, t0 = {}, time.time()
    for s in sorted(D):
        X = preprocess(D[s][0][:, idx].astype("float64"), smooth=(cond == "paper"))
        cls, avg = [], []
        with torch.no_grad():
            for i in range(0, len(X), 16):
                e = model.get_embedding(torch.from_numpy(X[i:i + 16]), sampling_rate=SR_NW, device=dev)  # bn,c,P,768
                cls.append(e[:, :, 0].cpu().numpy()); avg.append(e[:, :, 1:].mean(2).cpu().numpy())
        out[f"cls_{s}"], out[f"avg_{s}"], out[f"y_{s}"] = np.concatenate(cls), np.concatenate(avg), D[s][1]
        print(f"   [{cond}] subject {s}: {len(X)} trials ({time.time() - t0:.0f}s)", flush=True)
    np.savez(EMB(cond), **out)


def probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    meta = pickle.load(open(f"{ROOT}/heldout/_cache_cho2017_meta.pkl", "rb")); clean = meta["clean"]
    res = {}
    for cond in ("paper", "nosmooth"):
        if not os.path.exists(EMB(cond)): continue
        E = np.load(EMB(cond))
        for pool in ("cls", "avg"):
            accs = []
            for seed in SEEDS:                               # identical split code to step3d_matrix.py
                perm = np.random.RandomState(seed).permutation(clean)
                te, va, tr = list(perm[:10]), list(perm[10:15]), list(perm[15:])
                g = lambda ss: (np.concatenate([E[f"{pool}_{s}"].reshape(len(E[f"{pool}_{s}"]), -1) for s in ss]),
                                np.concatenate([E[f"y_{s}"] for s in ss]))
                (Xtr, ytr), (Xva, yva) = g(tr), g(va)
                best = max((make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(Xtr, ytr)
                            for C in (1e-4, 1e-3, 1e-2, 1e-1)), key=lambda m: m.score(Xva, yva))
                accs.append(float(np.mean([best.score(*g([s])) for s in te])))   # per-subject mean, as for EEGNet/LaBraM
            res[f"normwear_probe|{cond}|{pool}"] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                                                     "per_seed": accs}
            print(f"   {cond:9s} {pool}: {np.mean(accs) * 100:.1f} +- {np.std(accs) * 100:.1f}")
    json.dump({"channels": CHANS, "results": res}, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    self_check()
    {"embed": lambda: embed(sys.argv[2]), "probe": probe, "check": lambda: print("self_check ok")}[sys.argv[1]]()
