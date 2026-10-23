"""Subject-independent seizure detection on BIDS Siena, scored with SzCORE event-based metrics.

Question: does the LaBraM foundation model (pretrained on broadband EEG) transfer to
seizure detection on unseen patients better than EEGNet trained from scratch?

Usage (each fold runs in its own process so MPS memory is released, same as step3d):
  python seizure/szcore_siena.py prep                   # BIDS EDF -> per-subject 200 Hz float16 cache
  python seizure/szcore_siena.py run <EEGNet|fullFT> <fold>
  python seizure/szcore_siena.py summary

Protocol (SzCORE, Dan et al. 2024):
  - 19 channels, 10-20, common average (as stored in BIDS Siena), 0.5-75 Hz + 50 Hz notch, 200 Hz
  - subject-independent 7-fold CV: 2 test subjects per fold, 1 validation subject, rest train
  - event scoring with timescoring: any-overlap, 30 s pre / 60 s post tolerance,
    merge events < 90 s apart, split events > 5 min
"""
import os, sys, json, glob, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

ROOT = os.path.expanduser("~/bci-project")
# SZ_DATASET=chbmit: held-out replication on BIDS CHB-MIT, which is not in LaBraM's pretraining corpus
# (Siena is). Same pipeline; 60 Hz mains, 8 folds of 3 test subjects, chb01/chb21 (same person) kept together.
DATASET = os.environ.get("SZ_DATASET", "siena")
BIDS = os.path.expanduser("~/mne_data/BIDS_Siena" if DATASET == "siena" else "~/mne_data/BIDS_CHB-MIT")
CACHE = f"{ROOT}/seizure/_cache" + ("" if DATASET == "siena" else f"_{DATASET}")
MAINS = 50. if DATASET == "siena" else 60.
# SZ_SCHEME=loso: leave-one-patient-out (14 folds, 1 test / 2 validation / 11 training), the SzCORE
# recommendation for subject-independent models; default k7 keeps the original 7-fold results intact.
SCHEME = os.environ.get("SZ_SCHEME", "k7")
SUFFIX = ("" if DATASET == "siena" else f"_{DATASET}") + ("" if SCHEME == "k7" else f"_{SCHEME}")
OUT = f"{ROOT}/seizure/results_szcore_{DATASET}" + ("" if SCHEME == "k7" else f"_{SCHEME}") + ".json"
MODELS = f"{ROOT}/seizure/_models{SUFFIX}"
LABRAM_CACHE = os.path.expanduser("~/mne_data/labram_pretrained")
SFREQ, WIN_S, STRIDE_S = 200, 4, 2           # 4 s windows = 4 LaBraM patches of 200 samples
N_TIMES, PATCH = SFREQ * WIN_S, 200
CHANNELS = ["Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5", "Fz", "Cz", "Pz",
            "Fp2", "F4", "C4", "P4", "O2", "F8", "T4", "T6"]
if DATASET == "chbmit":     # CHB-MIT is stored as the 18-channel bipolar double banana; no referential signals exist
    CHANNELS = ["FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FZ-CZ",
                "CZ-PZ", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2"]
N_FOLDS, BG_RATIO = (14 if SCHEME == "loso" else 8 if DATASET == "chbmit" else 7), 5                      # background windows kept per seizure window in training


def subjects():
    return sorted(os.path.basename(p) for p in glob.glob(f"{BIDS}/sub-*"))


# ---------- prep: one cache file per subject (data + 1 Hz seizure mask) ----------
def do_prep():
    import mne, pandas as pd
    mne.set_log_level("ERROR")
    os.makedirs(CACHE, exist_ok=True)
    for sub in subjects():
        out = f"{CACHE}/{sub}.npz"
        if os.path.exists(out):
            continue
        xs, masks = [], []
        for edf in sorted(glob.glob(f"{BIDS}/{sub}/**/*_eeg.edf", recursive=True)):
            raw = mne.io.read_raw_edf(edf, preload=True)
            canon = {c.lower(): c for c in CHANNELS}           # some files use FP2 / CZ
            canon.update({"t7": "T3", "t8": "T4", "p7": "T5", "p8": "T6"})   # 10-10 names for the same sites
            raw.rename_channels(lambda c: canon.get(c.replace("-Avg", "").strip().lower(), c))
            try:
                raw.pick(CHANNELS).reorder_channels(CHANNELS)
            except ValueError:                                  # recording lacks part of the montage
                print(f"   skip {os.path.basename(edf)}: missing channels"); continue
            raw.notch_filter(MAINS).filter(0.5, 75.).resample(SFREQ)
            x = raw.get_data().astype("float32") * 1e6                 # V -> uV, as in step3d
            n_sec = x.shape[1] // SFREQ
            mask = np.zeros(n_sec, dtype=bool)
            ev = pd.read_csv(edf.replace("_eeg.edf", "_events.tsv"), sep="\t")
            for _, r in ev[ev["eventType"].astype(str).str.startswith("sz")].iterrows():
                mask[int(r["onset"]):int(np.ceil(r["onset"] + r["duration"]))] = True
            xs.append(x[:, :n_sec * SFREQ].astype("float16")); masks.append(mask)
        np.savez(out, x=np.concatenate(xs, 1), mask=np.concatenate(masks),
                 bounds=np.cumsum([0] + [m.size for m in masks]))
        print(f"   {sub}: {sum(m.size for m in masks)/3600:.1f} h, "
              f"{int(np.concatenate(masks).sum())} s seizure")


def load(sub):
    d = np.load(f"{CACHE}/{sub}.npz")
    return d["x"], d["mask"], d["bounds"]


def windows(x, mask, bounds, stride_s):
    """Window starts (in seconds) that do not cross a recording boundary, plus labels."""
    starts = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        starts.extend(range(a, b - WIN_S + 1, stride_s))
    starts = np.array(starts)
    y = np.array([mask[s:s + WIN_S].mean() >= 0.5 for s in starts], dtype="int64")
    return starts, y


def sample_train(subs, rng):
    """All seizure windows plus BG_RATIO x as many random background windows per subject."""
    X, Y = [], []
    for s in subs:
        x, m, b = load(s)
        st, y = windows(x, m, b, STRIDE_S)
        pos = np.where(y == 1)[0]
        neg = rng.choice(np.where(y == 0)[0], size=min((y == 0).sum(), max(BG_RATIO * len(pos), 200)),
                         replace=False)
        for i in np.concatenate([pos, neg]):
            X.append(x[:, st[i] * SFREQ:(st[i] + WIN_S) * SFREQ]); Y.append(y[i])
    return np.stack(X).astype("float32"), np.array(Y)


def subject_prob_trace(m, sub, dev, limit=None):
    """Seizure probability per second for a whole subject (overlapping windows averaged)."""
    import torch
    x, mask, b = load(sub)
    st, _ = windows(x, mask, b, STRIDE_S)
    if limit: st = st[:limit]
    m.eval(); acc = np.zeros(mask.size); cnt = np.zeros(mask.size)
    with torch.no_grad():
        for i in range(0, len(st), 64):
            xb = np.stack([x[:, s * SFREQ:(s + WIN_S) * SFREQ] for s in st[i:i + 64]]).astype("float32")
            p = torch.softmax(m(torch.tensor(xb).to(dev)), 1)[:, 1].cpu().numpy()
            for s, pi in zip(st[i:i + 64], p):
                acc[s:s + WIN_S] += pi; cnt[s:s + WIN_S] += 1
    return acc / np.maximum(cnt, 1), mask


def folds():
    if DATASET == "chbmit":
        return folds_chbmit()
    subs = subjects()
    rng = np.random.RandomState(0)
    perm = list(rng.permutation(subs))
    out = []
    n_test = 1 if SCHEME == "loso" else 2
    for k in range(N_FOLDS):
        test = perm[n_test * k:n_test * k + n_test]
        if SCHEME == "loso":                                      # validation = the next patients in the cycle
            rest = perm[n_test * k + n_test:] + perm[:n_test * k]
        else:                                                     # original 7-fold definition, unchanged
            rest = [s for s in perm if s not in test]
        out.append({"test": test, "val": rest[:2], "train": rest[2:]})
    return out


def folds_chbmit():
    """8 folds over 23 people (chb01 and chb21 are the same person and always share a fold)."""
    subs = subjects()
    units = [[s] for s in subs if s not in ("sub-01", "sub-21")]
    if "sub-01" in subs: units.append([u for u in ("sub-01", "sub-21") if u in subs])
    rng = np.random.RandomState(0)
    perm = [units[i] for i in rng.permutation(len(units))]
    out, k0 = [], 0
    sizes = [len(perm) // N_FOLDS + (1 if i < len(perm) % N_FOLDS else 0) for i in range(N_FOLDS)]
    for n in sizes:
        test_u = perm[k0:k0 + n]
        rest_u = perm[k0 + n:] + perm[:k0]
        k0 += n
        flat = lambda us: [s for u in us for s in u]
        out.append({"test": flat(test_u), "val": flat(rest_u[:2]), "train": flat(rest_u[2:])})
    return out


# ---------- event scoring ----------
THRESHOLDS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
SMOOTH_S = (5, 10, 20, 30)


def to_events(prob_1hz, thr, k=10):
    """Threshold a 1 Hz probability trace; keep only detections sustained for about k seconds."""
    det = prob_1hz >= thr
    smooth = np.convolve(det.astype(float), np.ones(k) / k, mode="same") >= 0.5
    return smooth


def score(ref_mask, hyp_mask):
    from timescoring import scoring
    from timescoring.annotations import Annotation
    params = scoring.EventScoring.Parameters(toleranceStart=30, toleranceEnd=60, minOverlap=0,
                                             maxEventDuration=5 * 60, minDurationBetweenEvents=90)
    ref, hyp = Annotation(ref_mask, 1), Annotation(hyp_mask, 1)
    s = scoring.EventScoring(ref, hyp, params)
    return {"tp": int(s.tp), "fp": int(s.fp), "refTrue": int(s.refTrue),
            "hours": float(ref_mask.size / 3600)}


def best_postproc(val_traces):
    """(threshold, smoothing seconds, F1) maximizing pooled event F1 over the validation subjects."""
    best = (0.5, 10, -1.0)
    for t in THRESHOLDS:
        for k in SMOOTH_S:
            f1 = aggregate([score(mask, to_events(p, t, k)) for p, mask in val_traces])["f1"]
            if f1 > best[2]: best = (t, k, f1)
    return best


def aggregate(rows):
    tp = sum(r["tp"] for r in rows); fp = sum(r["fp"] for r in rows)
    ref = sum(r["refTrue"] for r in rows); hours = sum(r["hours"] for r in rows)
    sens = tp / ref if ref else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * sens * prec / (sens + prec) if tp else 0.0      # no true detection -> F1 = 0, never NaN
    return {"sensitivity": sens, "precision": prec, "f1": f1, "fp_per_24h": fp / hours * 24,
            "n_seizures": ref, "hours": hours}


def build_model(kind):
    from braindecode.models import EEGNet, InterpolatedLaBraM
    if kind == "EEGNet":
        return EEGNet(n_chans=len(CHANNELS), n_outputs=2, n_times=N_TIMES, drop_prob=0.5)
    import mne
    info = mne.create_info(CHANNELS, SFREQ, "eeg")
    if DATASET == "chbmit":     # bipolar A-B placed at the midpoint of A and B, as braindecode does for LaBraM's own bipolar channels
        pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
        up = {k.upper(): v for k, v in pos.items()}
        for ch in info["chs"]:
            a, b = ch["ch_name"].split("-")
            ch["loc"][:3] = (up[a] + up[b]) / 2
    else:
        info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    full = InterpolatedLaBraM.from_pretrained("braindecode/labram-pretrained", chs_info=info["chs"],
            n_times=3000, n_outputs=2, patch_size=PATCH, strict=False, cache_dir=LABRAM_CACHE)
    mm = InterpolatedLaBraM(chs_info=info["chs"], n_times=N_TIMES, n_outputs=2, patch_size=PATCH)
    sf, sd, new = full.state_dict(), mm.state_dict(), {}
    for k, v in sd.items():
        if k in sf and sf[k].shape == v.shape: new[k] = sf[k]
        elif k == "temporal_embedding" and k in sf: new[k] = sf[k][:, :v.shape[1], :].clone()
        else: new[k] = v
    mm.load_state_dict(new, strict=False)
    return mm


# ---------- run one (model, fold) in its own process ----------
def load_results():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def do_run(kind, fold):
    fold = int(fold); key = f"{kind}|fold{fold}"
    if key in load_results() and not os.environ.get("FORCE"):
        print(f"   [{key}] done, skip"); return
    import torch, torch.nn as nn
    from braindecode.models import EEGNet, InterpolatedLaBraM
    DEV = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(fold); rng = np.random.RandomState(fold)
    F = folds()[fold]

    Xtr, ytr = sample_train(F["train"], rng)
    SMOKE = bool(os.environ.get("SMOKE"))
    if SMOKE:                                   # quick end-to-end check, not a result
        keep = np.concatenate([np.where(ytr == 1)[0][:64], np.where(ytr == 0)[0][:192]])
        Xtr, ytr = Xtr[keep], ytr[keep]
    print(f"   train windows {len(ytr)} (seizure {ytr.sum()})")

    def prob_trace(m, sub):
        return subject_prob_trace(m, sub, DEV, limit=600 if SMOKE else None)

    m = build_model(kind).to(DEV)
    lr, max_ep, bs = (1e-3, 40, 64) if kind == "EEGNet" else (1e-4, 15, 16)
    if SMOKE: max_ep = 1
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    lf = nn.CrossEntropyLoss()                 # v2: sampling is already 1:5; extra class weights made it trigger-happy
    best, best_state, patience, t0 = -1, None, 0, time.time()
    for ep in range(max_ep):
        m.train(); idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            opt.zero_grad()
            lf(m(torch.tensor(Xtr[j]).to(DEV)), torch.tensor(ytr[j]).to(DEV)).backward(); opt.step()
        vt = [prob_trace(m, s) for s in F["val"]]
        f1 = best_postproc(vt)[2]
        print(f"      ep{ep+1} val_F1={f1:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if best_state is None or f1 > best:
            best, patience = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            patience += 1
            if patience >= 4: break
    m.load_state_dict(best_state)
    if not SMOKE:                               # kept as the teacher for seizure/distill.py
        os.makedirs(MODELS, exist_ok=True); torch.save(best_state, f"{MODELS}/{kind}_fold{fold}.pt")
    vt = [prob_trace(m, s) for s in F["val"]]
    thr, k, _ = best_postproc(vt)                  # chosen on validation subjects only
    rows, traces = [], {}
    for s, (p, mask) in zip(F["val"], vt):
        traces[f"val_{s}_prob"], traces[f"val_{s}_mask"] = p.astype("float32"), mask
    for s in F["test"]:
        p, mask = prob_trace(m, s)
        traces[f"test_{s}_prob"], traces[f"test_{s}_mask"] = p.astype("float32"), mask
        r = score(mask, to_events(p, thr, k)); r["subject"] = s; rows.append(r)
    if not SMOKE:
        os.makedirs(MODELS, exist_ok=True); np.savez_compressed(f"{MODELS}/{kind}_fold{fold}_traces.npz", **traces)
    if SMOKE:
        print(f"   SMOKE OK [{key}] thr={thr} k={k} rows={rows}"); return
    res = load_results()
    res[key] = {"fold": fold, "test": F["test"], "val": F["val"], "threshold": thr, "smooth_s": k,
                "best_val_f1": best,
                "per_subject": rows, **aggregate(rows)}
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"   [{key}] F1={res[key]['f1']:.3f} sens={res[key]['sensitivity']:.3f} "
          f"FP/24h={res[key]['fp_per_24h']:.1f} ✓")


def do_summary():
    res = load_results(); final = {}
    for kind in ("EEGNet", "fullFT"):
        rows = [r for k, v in res.items() if k.startswith(kind + "|") for r in v["per_subject"]]
        if rows:
            final[kind] = aggregate(rows)
            final[kind]["n_folds"] = sum(k.startswith(kind + "|") for k in res)
            print(f"{kind:8s} folds={final[kind]['n_folds']} sens={final[kind]['sensitivity']:.3f} "
                  f"prec={final[kind]['precision']:.3f} F1={final[kind]['f1']:.3f} "
                  f"FP/24h={final[kind]['fp_per_24h']:.1f} seizures={final[kind]['n_seizures']}")
    res["summary"] = final
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "prep": do_prep()
    elif mode == "run": do_run(sys.argv[2], sys.argv[3])
    elif mode == "summary": do_summary()
