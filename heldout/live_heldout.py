"""Live held-out bench: datasets published AFTER every evaluated checkpoint was uploaded, same protocol as step3d_matrix.py.

Usage (one process per run, like step3d_matrix.py):
  LIVE_DATASET=ds008446 venv/bin/python heldout/live_heldout.py prep
  LIVE_DATASET=ds008446 venv/bin/python heldout/live_heldout.py run <kind> <seed>     # kind: EEGNet | probe | fullFT | cbramodProbe | cbramodFT
  LIVE_DATASET=ds008446 venv/bin/python heldout/live_heldout.py summary
Protocol (unchanged from the leaderboard): 0.1-75 Hz, 200 Hz, 1000-sample windows, subject-level random splits (seeds 0-3),
Adam + early stopping on held-out validation subjects, identical windows/splits/budget for every row.
Env: DEVICE=cpu forces CPU; SMOKE=1 runs 1 epoch on <=256 training windows and writes nothing.
"""
import os, sys, json, glob, pickle, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

ROOT = os.path.expanduser("~/bci-project")
DATA = os.path.expanduser("~/mne_data/live")
DS = os.environ.get("LIVE_DATASET", "ds008446")
SFREQ, N_TIMES, PATCH = 200, 1000, 200
BAND = (0.1, 75.)
SEEDS = [0, 1, 2, 3]
CACHE_HF = os.path.expanduser("~/mne_data/labram_pretrained")
OUT = f"{ROOT}/heldout/results_live_{DS}.json"
PKL = f"{ROOT}/heldout/_cache_live_{DS}.pkl"
SMOKE = os.environ.get("SMOKE") == "1"

# ---------------- dataset adapters: return {subject: (X[n, C, 1000] float32 uV, y int64)}, chs_info, class_names ----------------

def _epochs_to_arrays(raw, events, event_id, tmin, tmax, picks):
    import mne
    r = raw.copy().filter(*BAND, verbose="ERROR")
    ep = mne.Epochs(r, events, event_id, tmin=tmin, tmax=tmax, picks=picks, baseline=None, preload=True, verbose="ERROR")
    ep.resample(SFREQ, verbose="ERROR")
    X = ep.get_data()[:, :, :N_TIMES] * 1e6
    y = np.array([list(event_id.values()).index(v) for v in ep.events[:, -1]], dtype="int64")
    return X.astype("float32"), y, ep.info["chs"]


def ds008446_adapter():
    """UCL finger motor imagery (OpenNeuro ds008446, 2026-07-21). 20 subjects, g.tec 512 Hz, 62 EEG + 2 ref + Marker.
    Markers: 1 white screen, 2 focus cross, 3..7 thumb..pinky. Trials are read from marker transitions in the Marker channel.
    Task: 5-class finger MI (chance 20%). Window: cue onset -1 s .. +4 s like the MI leaderboard (truncated to 1000 samples @200 Hz)."""
    import mne
    store, chs_info, sd830 = {}, None, []
    for sdir in sorted(glob.glob(f"{DATA}/{DS}/sub-*")):
        sid = int(os.path.basename(sdir).split("-")[1])
        Xs, ys = [], []
        for f in sorted(glob.glob(f"{sdir}/eeg/*_eeg.edf")):
            raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
            # The EDF Marker channel is all zeros; labels live in *_events.tsv as one row per sample.
            # Rows start 10 s into the recording (onset column is sample/512^2, a units slip): sample = 5120 + row index.
            # Verified on complete files: EDF length 468992 samples = 463872 rows + 5120.
            rows = [l.split("\t") for l in open(f.replace("_eeg.edf", "_events.tsv"))][1:]
            tt = np.array([int(float(r[2])) for r in rows])
            if abs(raw.n_times - (len(tt) + 5120)) > 512:
                print(f"   skip {os.path.basename(f)}: EDF {raw.n_times} samples vs events {len(tt)}+5120 (incomplete file?)"); continue
            idx = np.where((np.diff(tt) != 0) & np.isin(tt[1:], [3, 4, 5, 6, 7]))[0] + 1
            on = idx + 5120
            eeg = [c for c in raw.ch_names if c not in ("Ref1", "Ref2", "Marker")]
            raw.pick(eeg)
            raw.set_montage("standard_1005", on_missing="ignore")
            # Recording-specific cleaning, same rule for every run: 50 Hz notch (UK mains; PhysioNet/Cho2017 were delivered
            # line-filtered), then channels whose 1-75 Hz SD exceeds 5x the run median or 300 uV are interpolated (many runs
            # have 15-20 railed electrodes), then common average reference. Counts are kept in the cache for the report.
            raw.notch_filter([50.], verbose="ERROR")
            sd = np.std(raw.copy().filter(1., 75., verbose="ERROR").get_data() * 1e6, axis=1)
            bad = [raw.ch_names[i] for i in np.where((sd > 5 * np.median(sd)) | (sd > 300.))[0]]
            if len(bad) > len(eeg) // 2:
                print(f"   skip {os.path.basename(f)}: {len(bad)}/{len(eeg)} channels bad"); continue
            raw.info["bads"] = bad
            if bad: raw.interpolate_bads(reset_bads=True, verbose="ERROR")
            raw.set_eeg_reference("average", projection=False, verbose="ERROR")
            BAD_LOG.append((os.path.basename(f), len(bad)))
            sd830.append(float(np.median(np.std(raw.copy().filter(8., 30., verbose="ERROR").get_data() * 1e6, axis=1))))
            ev = np.c_[on, np.zeros_like(on), tt[idx]]
            event_id = {"thumb": 3, "index": 4, "middle": 5, "ring": 6, "pinky": 7}
            X, y, chs = _epochs_to_arrays(raw, ev, event_id, -1.0, 4.0, "eeg")   # cue lasts 6 s; window = cue -1 .. +4 s as on the MI leaderboard
            Xs.append(X); ys.append(y); chs_info = chs_info or chs
        if Xs:
            store[sid] = (np.concatenate(Xs), np.concatenate(ys))
            print(f"   sub-{sid:02d}: {len(store[sid][1])} trials, class counts {np.bincount(store[sid][1], minlength=5).tolist()}", flush=True)
    global SCALE_8_30
    SCALE_8_30 = float(np.median(sd830)) if sd830 else None
    return store, chs_info, ["thumb", "index", "middle", "ring", "pinky"]


SCALE_8_30 = None          # dataset median 8-30 Hz channel SD (uV), filled by the adapter
BAD_LOG = []               # (file, n_interpolated_channels) per run, filled by the adapter
PHYSIONET_SD_8_30 = 12.70  # same reference constant as step3d_matrix.py: pretrained models see PhysioNet-range amplitudes


ADAPTERS = {"ds008446": ds008446_adapter}

# ---------------- prep ----------------

def do_prep():
    if os.path.exists(PKL):
        print("   cache exists, skipping prep"); return
    import mne; mne.set_log_level("ERROR")
    store, chs_info, classes = ADAPTERS[DS]()
    shapes = {s: v[0].shape[1:] for s, v in store.items()}
    ref = max(set(shapes.values()), key=list(shapes.values()).count)
    clean = sorted(s for s, sh in shapes.items() if sh == ref)
    dropped = sorted(set(store) - set(clean))
    pickle.dump({"store": {s: store[s] for s in clean}, "chs_info": chs_info, "classes": classes, "clean": clean,
                 "n_chans": int(ref[0]), "dropped": dropped, "sd_8_30_uV": SCALE_8_30, "bad_channels": BAD_LOG}, open(PKL, "wb"))
    nb = [n for _, n in BAD_LOG]
    print(f"   cached {len(clean)} subjects with shape {ref}; dropped {dropped}; median 8-30 Hz SD {SCALE_8_30:.1f} uV; "
          f"interpolated channels per run: median {np.median(nb):.0f}, max {max(nb)}")

# ---------------- run ----------------

def load_results():
    if os.path.exists(OUT):
        try:
            return json.load(open(OUT)).get("results", {})
        except Exception:
            pass
    return {}


def split_subjects(clean, seed):
    """Test ~25%, val ~15%, train the rest (at least 2 test, 2 val)."""
    rng = np.random.RandomState(seed); perm = list(rng.permutation(clean))
    n = len(perm); nt = max(2, round(0.25 * n)); nv = max(2, round(0.15 * n))
    return perm[:nt], perm[nt:nt + nv], perm[nt + nv:]


def do_run(kind, seed):
    seed = int(seed); key = kind
    res = load_results()
    if not SMOKE and any(r["seed"] == seed for r in res.get(key, [])):
        print(f"   [{kind} seed{seed}] done, skipping"); return
    import torch, torch.nn as nn
    sys.path.insert(0, f"{ROOT}/heldout"); import mps_sdpa  # noqa: F401
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import accuracy_score, cohen_kappa_score, balanced_accuracy_score
    from braindecode.models import EEGNet, InterpolatedLaBraM
    DEV = os.environ.get("DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")

    meta = pickle.load(open(PKL, "rb")); D, CHS_INFO, clean = meta["store"], meta["chs_info"], meta["clean"]
    n_cls, n_ch = len(meta["classes"]), meta["n_chans"]
    # One global amplitude constant per dataset, as for Cho2017 on the leaderboard: median 8-30 Hz channel SD -> PhysioNet's 12.7 uV.
    sc = np.float32(PHYSIONET_SD_8_30 / meta["sd_8_30_uV"]) if meta.get("sd_8_30_uV") else np.float32(1.0)
    D = {k: (X * sc, y) for k, (X, y) in D.items()}
    print(f"   amplitude scale x{float(sc):.3f}")
    test_s, val_s, train_s = split_subjects(clean, seed)
    gather = lambda subs: (np.concatenate([D[s][0] for s in subs]), np.concatenate([D[s][1] for s in subs]))
    Xtr, ytr = gather(train_s); Xva, yva = gather(val_s)
    if SMOKE: Xtr, ytr = Xtr[:256], ytr[:256]
    print(f"   {DS}: {n_ch} ch, {n_cls} classes | train {len(train_s)} subj/{len(ytr)} win, val {len(val_s)}, test {len(test_s)} | {DEV}")

    @torch.no_grad()
    def predict(m, X, bs=32):
        m.eval(); o = []
        for i in range(0, len(X), bs):
            o.append(m(torch.tensor(X[i:i + bs]).to(DEV)).argmax(1).cpu().numpy())
        return np.concatenate(o)

    def train(m, params, lr, wd, max_ep, patience, bs):
        m.to(DEV); opt = torch.optim.Adam(params, lr=lr, weight_decay=wd); lf = nn.CrossEntropyLoss()
        ld = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=bs, shuffle=True)
        best, state, pc, t0, last = -1, None, 0, time.time(), (0, 0)
        for ep in range(1 if SMOKE else max_ep):
            m.train()
            for xb, yb in ld:
                opt.zero_grad(); lf(m(xb.to(DEV)), yb.to(DEV)).backward(); opt.step()
            va = float(balanced_accuracy_score(yva, predict(m, Xva))); ta = float((predict(m, Xtr[:600]) == ytr[:600]).mean())
            last = (ta, va)
            if va > best: best, pc, state = va, 0, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            else: pc += 1
            if pc >= patience: break
        m.load_state_dict(state)
        print(f"      stop@ep{ep + 1} train={last[0]:.3f} best_val_bacc={best:.3f} ({time.time() - t0:.0f}s)")
        return m, last

    def labram():
        full = InterpolatedLaBraM.from_pretrained("braindecode/labram-pretrained", chs_info=CHS_INFO, n_times=3000,
                                                  n_outputs=n_cls, patch_size=PATCH, strict=False, cache_dir=CACHE_HF)
        mm = InterpolatedLaBraM(chs_info=CHS_INFO, n_times=N_TIMES, n_outputs=n_cls, patch_size=PATCH)
        sf, sd, new = full.state_dict(), mm.state_dict(), {}
        for k, v in sd.items():
            if k in sf and sf[k].shape == v.shape: new[k] = sf[k]
            elif k == "temporal_embedding" and k in sf: new[k] = sf[k][:, :v.shape[1], :].clone()
            else: new[k] = v
        mm.load_state_dict(new, strict=False); return mm

    torch.manual_seed(seed)
    if kind == "EEGNet":
        m = EEGNet(n_chans=n_ch, n_outputs=n_cls, n_times=N_TIMES, drop_prob=0.5)
        m, last = train(m, m.parameters(), 1e-3, 1e-4, 60, 10, 32)
    elif kind == "probe":
        m = labram()
        for n, p in m.named_parameters(): p.requires_grad = n.startswith("final_layer.")
        m, last = train(m, [p for p in m.parameters() if p.requires_grad], 1e-3, 1e-4, 40, 8, 32)
    elif kind == "fullFT":
        m = labram()
        m, last = train(m, m.parameters(), 1e-4, 1e-4, 30, 8, 16)
    elif kind in ("cbramodProbe", "cbramodFT"):
        from braindecode.models import CBraMod
        m = CBraMod.from_pretrained("braindecode/cbramod-pretrained", n_outputs=n_cls, n_chans=n_ch, n_times=N_TIMES, cache_dir=CACHE_HF)
        m.register_forward_pre_hook(lambda mod, inp: (inp[0] * 0.01,))   # CBraMod expects units of 100 uV
        probe = kind == "cbramodProbe"
        for n, p in m.named_parameters(): p.requires_grad = (not probe) or n.startswith("final_layer.")
        params = [p for p in m.parameters() if p.requires_grad]
        m, last = train(m, params, 1e-3, 1e-4, 40, 8, 32) if probe else train(m, params, 1e-4, 1e-4, 30, 8, 16)
    else:
        raise SystemExit(f"unknown kind {kind}")

    rows = []
    for s in test_s:
        Xs, ys = D[s]; pred = predict(m, Xs)
        rows.append({"subject": int(s), "acc": float(accuracy_score(ys, pred)), "bacc": float(balanced_accuracy_score(ys, pred)),
                     "kappa": float(cohen_kappa_score(ys, pred)), "n": int(len(ys))})
    macc = float(np.mean([r["acc"] for r in rows])); mb = float(np.mean([r["bacc"] for r in rows])); mk = float(np.mean([r["kappa"] for r in rows]))
    print(f"   [{kind} seed{seed}] => acc={macc * 100:.1f}% bacc={mb * 100:.1f}% kappa={mk:+.3f} (chance {100 / n_cls:.0f}%)")
    if SMOKE:
        print("   SMOKE: not written"); return
    res = load_results()
    res.setdefault(key, []).append({"seed": seed, "mean_acc": macc, "mean_bacc": mb, "mean_kappa": mk, "train_acc_last": last[0],
                                    "test_subjects": [int(s) for s in test_s], "per_subject": rows})
    json.dump({"dataset": DS, "results": res}, open(OUT, "w"), indent=2)
    print("   saved")

# ---------------- summary ----------------

def do_summary():
    res = load_results(); meta = pickle.load(open(PKL, "rb"))
    final = {}
    print(f"\n{DS}: {len(meta['clean'])} subjects, {meta['n_chans']} ch, classes {meta['classes']} (chance {100 / len(meta['classes']):.1f}%)")
    print(f"{'model':<14}{'acc (seeds)':<22}{'bal. acc':<22}{'kappa':<16}n")
    for kind in ["EEGNet", "probe", "fullFT", "cbramodProbe", "cbramodFT"]:
        rows = res.get(kind, [])
        if not rows: print(f"{kind:<14}(none)"); continue
        a = np.array([r["mean_acc"] for r in rows]); b = np.array([r["mean_bacc"] for r in rows]); k = np.array([r["mean_kappa"] for r in rows])
        final[kind] = {"acc_mean": float(a.mean()), "acc_std": float(a.std()), "bacc_mean": float(b.mean()), "bacc_std": float(b.std()),
                       "kappa_mean": float(k.mean()), "kappa_std": float(k.std()), "n_seeds": len(a)}
        print(f"{kind:<14}{a.mean() * 100:5.1f} ± {a.std() * 100:4.1f}%        {b.mean() * 100:5.1f} ± {b.std() * 100:4.1f}%        {k.mean():+.3f} ± {k.std():.3f}   {len(a)}")
    json.dump({"dataset": DS, "final_mean_std_over_seeds": final, "results": res}, open(OUT, "w"), indent=2)
    print("SUMMARY_DONE")


if __name__ == "__main__":
    mode = sys.argv[1]
    {"prep": lambda: do_prep(), "run": lambda: do_run(sys.argv[2], sys.argv[3]), "summary": lambda: do_summary()}[mode]()
