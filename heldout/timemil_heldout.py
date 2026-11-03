"""TimeMIL (Chen et al., ICML 2024) trained from scratch under the held-out-subject protocol of step3d_matrix.py.

Usage (one run per process, as in step3d_matrix.py, so MPS memory is released between runs):
  MI_DATASET=cho2017   python heldout/timemil_heldout.py run broad_0.1-75 0
  MI_DATASET=cho2017   python heldout/timemil_heldout.py all broad_0.1-75      # seeds 0-3 in subprocesses, then summary
  MI_DATASET=physionet python heldout/timemil_heldout.py all broad_0.1-75
  MI_DATASET=...       python heldout/timemil_heldout.py summary
Env: DEVICE=cpu forces CPU; SMOKE=1 runs 1 epoch on 256 training trials and writes nothing.

Protocol: everything that can be imported from step3d_matrix.py is imported (dataset selection, caches, META,
Cho2017 amplitude scale, SEEDS, N_TIMES). The subject split and train()/predict() are closures inside
step3d_matrix.do_run (lines 138-166) and cannot be imported, so they are copied verbatim below.
Model: official TimeMIL code (external/TimeMIL, commit 7542c1c) with its default architecture hyperparameters
(InceptionTime extractor out_channels=32 -> mDim=128, 2 Nystrom TransLayers, 8 heads, dropout 0.2), plus a
numerical floor on the learnable wavelet scale (see build_timemil; without it training produced NaN at epoch 3).
Optimiser, loss, batch size, max epochs and early stopping are the EEGNet settings of step3d_matrix.py line 182.
Output: heldout/results_<dataset>_timemil.json (new file; existing result files are never touched).
"""
import os, sys, json, pickle, time, subprocess

ROOT = os.path.expanduser("~/bci-project")
sys.path.insert(0, ROOT)
import step3d_matrix as P                      # noqa: E402  (reads MI_DATASET at import; no side effects)
import numpy as np                             # noqa: E402

OUT = f"{ROOT}/heldout/results_{P.DATASET}_timemil.json"
KIND = "TimeMIL"
SMOKE = os.environ.get("SMOKE") == "1"


def load_results():
    if os.path.exists(OUT):
        try:
            return json.load(open(OUT)).get("results", {})
        except Exception:
            pass
    return {}


SCALE_FLOOR = 1e-2   # floor on |scale| of the learnable Mexican-hat wavelets (see build_timemil)


def build_timemil(n_chans, n_times, dropout=0.2, mDim=128):
    """Official TimeMIL with its default hyperparameters. Two changes to the wavelet kernel function only:
    (1) the hard-coded .cuda() is replaced by device-aware placement;
    (2) |scale| is floored at SCALE_FLOOR. The official kernel is even in `scale` and singular at 0
        (1/|scale|^0.5 and (x/scale)^2); its init `scale = 1 + N(0,1)` ("make sure scale > 0" in the source) does
        not keep it away from 0, and under Adam lr=1e-3 one scale reached 1e-17 at epoch 3 (Cho2017 seed 0) and
        produced NaN gradients -> constant predictions. For |scale| >= SCALE_FLOOR the function is unchanged; below
        it the wavelet is already a delta on the 19-tap integer grid, so the floor only removes the singularity."""
    import torch, torch.nn as nn
    import sys as _sys, os as _os; _sys.path.insert(0, _os.path.expanduser("~/bci-project/heldout")); import mps_groupconv  # noqa  MPS grouped-conv bug fix (docs/mps_groupconv_bug.md)
    sys.path.insert(0, f"{ROOT}/external/TimeMIL")
    import models.timemil as TM

    def mexican_hat_wavelet(size, scale, shift):      # == TM.mexican_hat_wavelet minus .cuda(), plus the floor
        x = torch.linspace(-(size[1] - 1) // 2, (size[1] - 1) // 2, size[1], device=scale.device)
        x = x.reshape(1, -1).repeat(size[0], 1) - shift
        scale = scale.abs().clamp_min(SCALE_FLOOR)
        C = 2 / (3 ** 0.5 * torch.pi ** 0.25)
        return C * (1 - (x / scale) ** 2) * torch.exp(-(x / scale) ** 2 / 2) * 1 / (torch.abs(scale) ** 0.5)
    TM.mexican_hat_wavelet = mexican_hat_wavelet

    class ChansFirstToTimeMIL(nn.Module):
        """Our windows are [B, 64 ch, 1000 t]; TimeMIL expects [B, length, channels]."""
        def __init__(self):
            super().__init__()
            self.m = TM.TimeMIL(in_features=n_chans, n_classes=2, mDim=mDim, max_seq_len=n_times, dropout=dropout)

        def forward(self, x):
            return self.m(x.transpose(1, 2))
    return ChansFirstToTimeMIL()


def count_params(m):
    ps = [p for p in m.parameters() if p.requires_grad]
    nominal = sum(p.numel() for p in ps)
    unique = sum(p.numel() for p in {p.data_ptr(): p for p in ps}.values())   # wave*_ share storage with wave* in the official code
    return nominal, unique


def do_run(band, seed):
    seed = int(seed); key = f"{KIND}|{band}"
    if not SMOKE and any(r["seed"] == seed for r in load_results().get(key, [])):
        print(f"   [{KIND}/{band} seed{seed}] already done, skip"); return
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import accuracy_score, cohen_kappa_score
    DEV = os.environ.get("DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")
    t_run = time.time()

    # ---- data, scale and split: step3d_matrix.py lines 132-142, verbatim ----
    meta = pickle.load(open(P.META, "rb")); clean = meta["clean"]
    D = pickle.load(open(P.band_cache(band), "rb"))
    if P.DATASET == "cho2017":
        sc = float(P.CHO_SCALE) if P.CHO_SCALE else P.CHO_SCALE_DEFAULT
        D = {k: (X * np.float32(sc), y) for k, (X, y) in D.items()}
        print(f"   cho2017 amplitude scale x{sc:.4f}")
    rng = np.random.RandomState(seed); perm = rng.permutation(clean)
    test_s, val_s, train_s = list(perm[:10]), list(perm[10:15]), list(perm[15:])
    def gather(subs): return (np.concatenate([D[s][0] for s in subs], 0),
                              np.concatenate([D[s][1] for s in subs], 0))
    Xtr, ytr = gather(train_s); Xva, yva = gather(val_s)
    if SMOKE:
        Xtr, ytr = Xtr[:256], ytr[:256]
    print(f"   {P.DATASET}/{band} seed{seed} dev={DEV} train={len(Xtr)} val={len(Xva)} "
          f"test_subjects={[int(s) for s in test_s]}", flush=True)

    # ---- predict() and train(): step3d_matrix.py lines 144-166, verbatim except the two bookkeeping lines marked + ----
    @torch.no_grad()
    def predict(m, X, bs=32):
        m.eval(); o = []
        for i in range(0, len(X), bs):
            o.append(m(torch.tensor(X[i:i+bs]).to(DEV)).argmax(1).cpu().numpy())
        return np.concatenate(o)

    def train(m, params, lr, wd, max_ep, patience, bs):
        m.to(DEV); opt = torch.optim.Adam(params, lr=lr, weight_decay=wd); lf = nn.CrossEntropyLoss()
        ld = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=bs, shuffle=True)
        best, bs_state, pc, t0, last = -1, None, 0, time.time(), (0, 0)
        best_ep = 0                                                                         # +
        for ep in range(max_ep):
            m.train()
            for xb, yb in ld:
                opt.zero_grad(); lf(m(xb.to(DEV)), yb.to(DEV)).backward(); opt.step()
            va = float((predict(m, Xva) == yva).mean()); ta = float((predict(m, Xtr[:600]) == ytr[:600]).mean())
            last = (ta, va)
            if va > best: best, pc, bs_state = va, 0, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            else: pc += 1
            if va == best and pc == 0: best_ep = ep + 1                                     # +
            print(f"      ep{ep+1:02d} train={ta:.3f} val={va:.3f} best={best:.3f} pc={pc} ({(time.time()-t0)/60:.1f} min)", flush=True)
            if pc >= patience: break
        m.load_state_dict(bs_state)
        print(f"      停@ep{ep+1} train={last[0]:.3f} best_val={best:.3f} ({time.time()-t0:.0f}s)")
        train.info = {"epochs_run": ep + 1, "best_epoch": best_ep, "best_val_acc": best}    # +
        return m, last

    torch.manual_seed(seed)                                                  # step3d_matrix.py line 179
    m = build_timemil(n_chans=Xtr.shape[1], n_times=P.N_TIMES)              # TimeMIL defaults
    n_nom, n_uniq = count_params(m)
    print(f"   TimeMIL trainable params: {n_nom:,} (unique storage {n_uniq:,})", flush=True)
    HP = dict(lr=1e-3, wd=1e-4, max_ep=1 if SMOKE else 60, patience=10, bs=32)   # == EEGNet, step3d_matrix.py line 182
    m, last = train(m, m.parameters(), **HP)

    wave_scales = torch.cat([p.detach()[0].abs().flatten().cpu() for n, p in m.named_parameters() if ".wave" in n])
    diag = {"params_finite": bool(all(torch.isfinite(p).all() for p in m.parameters())),
            "min_abs_wave_scale": float(wave_scales.min()), "n_wave_scales_at_floor": int((wave_scales <= SCALE_FLOOR).sum()),
            "n_wave_scales": int(wave_scales.numel())}
    print(f"   diag: {diag}", flush=True)
    rows = []
    for s in test_s:
        Xs, ys = D[s]; pred = predict(m, Xs)
        rows.append({"subject": int(s), "acc": float(accuracy_score(ys, pred)),
                     "kappa": float(cohen_kappa_score(ys, pred))})
    macc = float(np.mean([r["acc"] for r in rows])); mkap = float(np.mean([r["kappa"] for r in rows]))
    minutes = (time.time() - t_run) / 60
    print(f"   [{KIND}/{band} seed{seed}] => acc={macc*100:.1f}% kappa={mkap:+.3f} train={last[0]:.3f} "
          f"epochs={train.info['epochs_run']} best_ep={train.info['best_epoch']} {minutes:.1f} min", flush=True)
    if SMOKE:
        print("   SMOKE run: nothing written"); return
    res = load_results()
    res.setdefault(key, []).append({"seed": seed, "mean_acc": macc, "mean_kappa": mkap, "train_acc_last": last[0],
                                    "per_subject": rows, **train.info, **diag, "minutes": minutes, "device": DEV,
                                    "n_params": n_nom, "n_params_unique_storage": n_uniq,
                                    "hparams": {**HP, "optimizer": "Adam", "loss": "CrossEntropy",
                                                "model": "TimeMIL default: InceptionTime out_channels=32, mDim=128, "
                                                         "2 TransLayers, 8 heads, Nystrom landmarks 64, dropout 0.2",
                                                "wavelet_scale_floor": SCALE_FLOOR,
                                                "input_scale": (float(P.CHO_SCALE) if P.CHO_SCALE else P.CHO_SCALE_DEFAULT)
                                                               if P.DATASET == "cho2017" else 1.0}})
    json.dump({"results": res}, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"   saved -> {OUT}", flush=True)


def do_summary():
    res = load_results(); final = {}
    for key, rows in res.items():
        a = np.array([r["mean_acc"] for r in rows]); k = np.array([r["mean_kappa"] for r in rows])
        final[key] = {"acc_mean": float(a.mean()), "acc_std": float(a.std()), "kappa_mean": float(k.mean()),
                      "kappa_std": float(k.std()), "n_seeds": len(a),
                      "epochs_run": [r.get("epochs_run") for r in rows], "best_epoch": [r.get("best_epoch") for r in rows],
                      "minutes": [round(r.get("minutes", 0), 1) for r in rows],
                      "per_seed_acc": [round(float(x) * 100, 2) for x in a]}
        print(f"{key:<24} {a.mean()*100:5.1f}% ± {a.std()*100:4.1f}% (n={len(a)}) kappa {k.mean():+.3f}±{k.std():.3f} "
              f"per-seed {final[key]['per_seed_acc']} epochs {final[key]['epochs_run']} min {final[key]['minutes']}")
    json.dump({"final_mean_std_over_seeds": final, "results": res}, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"saved -> {OUT}\nSUMMARY_DONE")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "run": do_run(sys.argv[2], sys.argv[3])
    elif mode == "summary": do_summary()
    elif mode == "all":
        band = sys.argv[2]
        for seed in P.SEEDS:
            r = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "run", band, str(seed)])
            if r.returncode != 0:
                print(f"   seed {seed} exited with {r.returncode}", flush=True)
        do_summary()
