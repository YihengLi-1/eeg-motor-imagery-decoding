"""完整矩阵(进程隔离版,根治 MPS 显存累积被 OS 杀进程的问题)。
用法(由 driver 循环调用):
  python step3d_matrix.py prep                 # 预处理两个频段,缓存到磁盘(只做一次)
  python step3d_matrix.py run <kind> <band> <seed>   # 一个全新进程只跑一个配置
  python step3d_matrix.py summary              # 汇总成最终表
每个 run 都是独立进程 -> 退出即释放全部显存,不累积。结果增量写入,断点续跑跳过已完成的。
防 leakage 四条不变。
"""
import os, sys, json, warnings, pickle, time
warnings.filterwarnings("ignore")
import numpy as np

ROOT = os.path.expanduser("~/bci-project")
OUT = f"{ROOT}/step3c_fullmatrix_results.json"        # 沿用已有结果文件(seed0 的 4 个配置已在里面)
META = f"{ROOT}/_cache_meta.pkl"
BANDS = {"narrow_8-30": (8., 30.), "broad_0.1-75": (0.1, 75.)}
SFREQ, N_TIMES, PATCH = 200, 1000, 200
SEEDS = [0, 1, 2, 3]
SUBJECTS, RUNS = list(range(1, 51)), [4, 8, 12]
CACHE = os.path.expanduser("~/mne_data/labram_pretrained")
def band_cache(b): return f"{ROOT}/_cache_{b}.pkl"

# ---------- 模式 1: 预处理并缓存 ----------
def do_prep():
    if os.path.exists(META) and all(os.path.exists(band_cache(b)) for b in BANDS):
        print("   缓存已存在,跳过预处理"); return
    import mne; mne.set_log_level("ERROR")
    from mne.datasets import eegbci
    from mne.io import read_raw_edf, concatenate_raws
    from mne import Epochs, pick_types, events_from_annotations
    print("   预处理两个频段(只做一次)...")
    store = {b: {} for b in BANDS}; chs_info = None
    for s in SUBJECTS:
        try:
            raw = concatenate_raws([read_raw_edf(f, preload=True, verbose="ERROR")
                                    for f in eegbci.load_data(subjects=[s], runs=RUNS)])
            eegbci.standardize(raw); raw.set_montage("standard_1005", on_missing="ignore")
            events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3), verbose="ERROR")
            picks = pick_types(raw.info, eeg=True, meg=False, stim=False, eog=False, exclude="bads")
            for bname, (lo, hi) in BANDS.items():
                r = raw.copy().filter(lo, hi, verbose="ERROR")
                ep = Epochs(r, events, dict(T1=2, T2=3), tmin=-1., tmax=4., picks=picks,
                            baseline=None, preload=True, verbose="ERROR")
                ep.resample(SFREQ, verbose="ERROR")
                X = ep.get_data()[:, :, :N_TIMES] * 1e6
                if X.shape[1] == 64 and X.shape[2] == N_TIMES:
                    store[bname][s] = (X.astype("float32"), (ep.events[:, -1] - 2).astype("int64"))
                    if chs_info is None: chs_info = ep.info["chs"]
        except Exception:
            continue
    clean = sorted(set(store["narrow_8-30"]) & set(store["broad_0.1-75"]))
    for b in BANDS:
        pickle.dump(store[b], open(band_cache(b), "wb"))
    pickle.dump({"chs_info": chs_info, "clean": clean}, open(META, "wb"))
    print(f"   缓存完成,两频段都干净的被试 {len(clean)} 个")

# ---------- 模式 2: 跑一个配置(独立进程) ----------
def load_results():
    if os.path.exists(OUT):
        try:
            d = json.load(open(OUT)); return d.get("results") or d.get("raw") or {}
        except Exception: pass
    return {}

def do_run(kind, band, seed):
    seed = int(seed); key = f"{kind}|{band}"
    res = load_results()
    if any(r["seed"] == seed for r in res.get(key, [])):
        print(f"   [{kind}/{band} seed{seed}] 已完成,跳过"); return
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import accuracy_score, cohen_kappa_score
    from braindecode.models import EEGNet, InterpolatedLaBraM
    DEV = "mps" if torch.backends.mps.is_available() else "cpu"

    meta = pickle.load(open(META, "rb")); CHS_INFO, clean = meta["chs_info"], meta["clean"]
    D = pickle.load(open(band_cache(band), "rb"))
    rng = np.random.RandomState(seed); perm = rng.permutation(clean)
    test_s, val_s, train_s = list(perm[:10]), list(perm[10:15]), list(perm[15:])
    def gather(subs): return (np.concatenate([D[s][0] for s in subs], 0),
                              np.concatenate([D[s][1] for s in subs], 0))
    Xtr, ytr = gather(train_s); Xva, yva = gather(val_s)

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
        for ep in range(max_ep):
            m.train()
            for xb, yb in ld:
                opt.zero_grad(); lf(m(xb.to(DEV)), yb.to(DEV)).backward(); opt.step()
            va = float((predict(m, Xva) == yva).mean()); ta = float((predict(m, Xtr[:600]) == ytr[:600]).mean())
            last = (ta, va)
            if va > best: best, pc, bs_state = va, 0, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            else: pc += 1
            if pc >= patience: break
        m.load_state_dict(bs_state)
        print(f"      停@ep{ep+1} train={last[0]:.3f} best_val={best:.3f} ({time.time()-t0:.0f}s)")
        return m, last

    def transplant():
        full = InterpolatedLaBraM.from_pretrained("braindecode/labram-pretrained", chs_info=CHS_INFO,
                n_times=3000, n_outputs=2, patch_size=PATCH, strict=False, cache_dir=CACHE)
        mm = InterpolatedLaBraM(chs_info=CHS_INFO, n_times=N_TIMES, n_outputs=2, patch_size=PATCH)
        sf, sd, new = full.state_dict(), mm.state_dict(), {}
        for k, v in sd.items():
            if k in sf and sf[k].shape == v.shape: new[k] = sf[k]
            elif k == "temporal_embedding" and k in sf: new[k] = sf[k][:, :v.shape[1], :].clone()
            else: new[k] = v
        mm.load_state_dict(new, strict=False); return mm

    torch.manual_seed(seed)
    if kind == "EEGNet":
        m = EEGNet(n_chans=64, n_outputs=2, n_times=N_TIMES, drop_prob=0.5)
        m, last = train(m, m.parameters(), 1e-3, 1e-4, 60, 10, 32)
    elif kind == "probe":
        m = transplant()
        for n, p in m.named_parameters(): p.requires_grad = n.startswith("final_layer.")
        m, last = train(m, [p for p in m.parameters() if p.requires_grad], 1e-3, 1e-4, 40, 8, 32)
    else:  # fullFT
        m = transplant()
        for p in m.parameters(): p.requires_grad = True
        m, last = train(m, m.parameters(), 1e-4, 1e-4, 30, 8, 16)

    rows = []
    for s in test_s:
        Xs, ys = D[s]; pred = predict(m, Xs)
        rows.append({"subject": int(s), "acc": float(accuracy_score(ys, pred)),
                     "kappa": float(cohen_kappa_score(ys, pred))})
    macc = float(np.mean([r["acc"] for r in rows])); mkap = float(np.mean([r["kappa"] for r in rows]))

    res = load_results()                       # 重新读,避免覆盖并发写入
    res.setdefault(key, []).append({"seed": seed, "mean_acc": macc, "mean_kappa": mkap,
                                    "train_acc_last": last[0], "per_subject": rows})
    json.dump({"results": res}, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"   [{kind}/{band} seed{seed}] => acc={macc*100:.1f}% kappa={mkap:+.3f} train={last[0]:.3f} ✓已存")

# ---------- 模式 3: 汇总 ----------
def do_summary():
    res = load_results()
    CONFIGS = [("EEGNet", "narrow_8-30"), ("EEGNet", "broad_0.1-75"),
               ("probe", "narrow_8-30"), ("probe", "broad_0.1-75"),
               ("fullFT", "narrow_8-30"), ("fullFT", "broad_0.1-75")]
    lab = {"EEGNet": "EEGNet端到端", "probe": "LaBraM探针", "fullFT": "LaBraM全量FT"}
    bl = {"narrow_8-30": "8-30Hz", "broad_0.1-75": "0.1-75Hz"}
    print("\n" + "=" * 80)
    print(f"{'模型':<14}{'频段':<12}{'Accuracy(对seed)':<22}{'Kappa':<18}{'末轮train':<10}")
    print("-" * 80)
    final = {}
    for kind, band in CONFIGS:
        rows = res.get(f"{kind}|{band}", [])
        if not rows:
            print(f"{lab[kind]:<12}{bl[band]:<10}  (无)"); continue
        a = np.array([r["mean_acc"] for r in rows]); k = np.array([r["mean_kappa"] for r in rows])
        tr = np.array([r["train_acc_last"] for r in rows])
        final[f"{kind}|{band}"] = {"acc_mean": float(a.mean()), "acc_std": float(a.std()),
                                   "kappa_mean": float(k.mean()), "kappa_std": float(k.std()),
                                   "train_acc_mean": float(tr.mean()), "n_seeds": len(a)}
        print(f"{lab[kind]:<12}{bl[band]:<10}{a.mean()*100:5.1f}% ± {a.std()*100:4.1f}% (n={len(a)})   "
              f"{k.mean():+.3f} ± {k.std():.3f}   {tr.mean():.3f}")
    print(f"\n{'随机猜测':<12}{'':<10} 50.0%")
    json.dump({"final_mean_std_over_seeds": final, "results": res},
              open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"\n已存 {OUT}\nSUMMARY_DONE")

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "prep": do_prep()
    elif mode == "run": do_run(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == "summary": do_summary()
