"""第三步最终:完整矩阵 = {EEGNet, LaBraM探针, LaBraM全量微调} × {窄带8-30, 宽带0.1-75} × 4 seed。
每个 seed 重新按被试切(35/5/10),三模型同一划分。报 mean±std(对 seed)。
防 leakage 四条不变:按被试切;同 seed 内三模型同划分;早停只用验证被试;测试只碰一次;只用 ×1e6。
增量存盘:每个 seed 跑完存一次,中途崩了也保留已完成的。
"""
import os, json, time, warnings, gc, traceback
warnings.filterwarnings("ignore")
import numpy as np
import mne; mne.set_log_level("ERROR")
from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws
from mne import Epochs, pick_types, events_from_annotations
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score
from braindecode.models import EEGNet, InterpolatedLaBraM

SUBJECTS, RUNS = list(range(1, 51)), [4, 8, 12]
SFREQ, N_TIMES, PATCH = 200, 1000, 200
SEEDS = [0, 1, 2, 3]
BANDS = {"narrow_8-30": (8., 30.), "broad_0.1-75": (0.1, 75.)}
CACHE = os.path.expanduser("~/mne_data/labram_pretrained")
OUT = os.path.expanduser("~/bci-project/step3c_fullmatrix_results.json")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"设备={DEV} | 完整矩阵: 3 模型 × 2 频段 × {len(SEEDS)} seed\n")

# ===== 1. 载入两个频段各一份(raw 只读一次,过两次滤波) =====
print("[1] 载入 + 预处理(两个频段)...")
data = {b: {} for b in BANDS}
CHS_INFO = None
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
                data[bname][s] = (X.astype("float32"), (ep.events[:, -1] - 2).astype("int64"))
                if CHS_INFO is None: CHS_INFO = ep.info["chs"]
    except Exception:
        continue
clean = sorted(set(data["narrow_8-30"]) & set(data["broad_0.1-75"]))
print(f"   两频段都干净的被试: {len(clean)} 个")

# ===== 工具 =====
@torch.no_grad()
def predict(model, X, bs=32):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.tensor(X[i:i+bs]).to(DEV)).argmax(1).cpu().numpy())
    return np.concatenate(out)

def train(model, trainable, Xtr, ytr, Xva, yva, lr, wd, max_ep, patience, bs, tag):
    model.to(DEV)
    opt = torch.optim.Adam(trainable, lr=lr, weight_decay=wd); lossfn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=bs, shuffle=True)
    best_val, best_state, pc, t0, last = -1, None, 0, time.time(), (0, 0)
    for ep in range(max_ep):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); lossfn(model(xb.to(DEV)), yb.to(DEV)).backward(); opt.step()
        va = float((predict(model, Xva) == yva).mean())
        ta = float((predict(model, Xtr[:600]) == ytr[:600]).mean())
        last = (ta, va)
        if va > best_val:
            best_val, pc = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: pc += 1
        if pc >= patience: break
    model.load_state_dict(best_state)
    print(f"      [{tag}] 停@ep{ep+1} train={last[0]:.3f} best_val={best_val:.3f} ({time.time()-t0:.0f}s)")
    return model, last  # last=(train_acc, val_acc) 末轮,用来看过拟合

def eval_subj(model, test_dict):
    rows = []
    for s, (Xs, ys) in test_dict.items():
        pred = predict(model, Xs)
        rows.append({"subject": int(s), "acc": float(accuracy_score(ys, pred)),
                     "kappa": float(cohen_kappa_score(ys, pred))})
    return rows

def transplant():
    full = InterpolatedLaBraM.from_pretrained("braindecode/labram-pretrained", chs_info=CHS_INFO,
            n_times=3000, n_outputs=2, patch_size=PATCH, strict=False, cache_dir=CACHE)
    m = InterpolatedLaBraM(chs_info=CHS_INFO, n_times=N_TIMES, n_outputs=2, patch_size=PATCH)
    sf, sd, new = full.state_dict(), m.state_dict(), {}
    for k, v in sd.items():
        if k in sf and sf[k].shape == v.shape: new[k] = sf[k]
        elif k == "temporal_embedding" and k in sf: new[k] = sf[k][:, :v.shape[1], :].clone()
        else: new[k] = v
    m.load_state_dict(new, strict=False)
    del full; gc.collect()
    return m

def build_and_train(kind, Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    if kind == "EEGNet":
        m = EEGNet(n_chans=64, n_outputs=2, n_times=N_TIMES, drop_prob=0.5)
        return train(m, m.parameters(), Xtr, ytr, Xva, yva, 1e-3, 1e-4, 60, 10, 32, kind)
    if kind == "probe":
        m = transplant()
        for n, p in m.named_parameters(): p.requires_grad = n.startswith("final_layer.")
        return train(m, [p for p in m.parameters() if p.requires_grad], Xtr, ytr, Xva, yva, 1e-3, 1e-4, 40, 8, 32, kind)
    if kind == "fullFT":
        m = transplant()
        for p in m.parameters(): p.requires_grad = True
        return train(m, m.parameters(), Xtr, ytr, Xva, yva, 1e-4, 1e-4, 30, 8, 16, kind)

# ===== 2. 跑矩阵 =====
CONFIGS = [("EEGNet", "narrow_8-30"), ("EEGNet", "broad_0.1-75"),
           ("probe", "narrow_8-30"), ("probe", "broad_0.1-75"),
           ("fullFT", "narrow_8-30"), ("fullFT", "broad_0.1-75")]
allres = {f"{k}|{b}": [] for k, b in CONFIGS}
# 断点续跑:载入之前已存的结果,跳过已完成的 (config, seed)
if os.path.exists(OUT):
    try:
        _s = json.load(open(OUT)); _s = _s.get("results") or _s.get("raw") or {}
        for _k in allres:
            if _k in _s: allres[_k] = _s[_k]
        print("   断点续跑,已载入: " + ", ".join(f"{k}={len(v)}seed" for k, v in allres.items() if v))
    except Exception:
        pass

def gather(D, subs):
    return (np.concatenate([D[s][0] for s in subs], 0), np.concatenate([D[s][1] for s in subs], 0))

for seed in SEEDS:
    rng = np.random.RandomState(seed); perm = rng.permutation(clean)
    test_s, val_s, train_s = list(perm[:10]), list(perm[10:15]), list(perm[15:])
    print(f"\n========== SEED {seed} | 测试被试 {sorted(int(s) for s in test_s)} ==========")
    for kind, band in CONFIGS:
        D = data[band]
        Xtr, ytr = gather(D, train_s); Xva, yva = gather(D, val_s)
        test_dict = {s: D[s] for s in test_s}
        tag = f"{kind}/{band}"
        if any(r["seed"] == seed for r in allres[f"{kind}|{band}"]):
            print(f"   --- {tag} --- (seed {seed} 已完成,跳过)"); continue
        try:
            print(f"   --- {tag} ---")
            m, (tr_acc, va_acc) = build_and_train(kind, Xtr, ytr, Xva, yva, seed)
            rows = eval_subj(m, test_dict)
            macc = float(np.mean([r["acc"] for r in rows])); mkap = float(np.mean([r["kappa"] for r in rows]))
            allres[f"{kind}|{band}"].append({"seed": seed, "mean_acc": macc, "mean_kappa": mkap,
                                             "train_acc_last": tr_acc, "val_acc_best_proxy": va_acc,
                                             "per_subject": rows})
            print(f"      => 测试 mean_acc={macc*100:.1f}% mean_kappa={mkap:+.3f}  (末轮 train={tr_acc:.3f},看过拟合)")
            m.to("cpu"); del m
        except Exception as e:
            print(f"      !! {tag} 失败: {type(e).__name__}: {str(e)[:150]}")
            traceback.print_exc()
        gc.collect()
        if DEV == "mps": torch.mps.empty_cache()
        json.dump({"results": allres}, open(OUT, "w"), indent=2, ensure_ascii=False)  # 每个配置存一次
    print(f"   [seed {seed} 完成]")

# ===== 3. 汇总(对 seed 求 mean±std) =====
def agg(key):
    accs = np.array([r["mean_acc"] for r in allres[key]])
    kaps = np.array([r["mean_kappa"] for r in allres[key]])
    trains = np.array([r["train_acc_last"] for r in allres[key]])
    return accs, kaps, trains

print("\n" + "=" * 78)
print(f"============ 完整矩阵最终结果(对 {len(SEEDS)} 个 seed 求 mean±std)============")
print(f"{'模型':<16}{'频段':<14}{'Accuracy':<18}{'Kappa':<16}{'末轮train(过拟合)':<12}")
print("-" * 78)
label = {"EEGNet": "EEGNet端到端", "probe": "LaBraM探针", "fullFT": "LaBraM全量FT"}
bl = {"narrow_8-30": "8-30Hz", "broad_0.1-75": "0.1-75Hz"}
final = {}
for kind, band in CONFIGS:
    key = f"{kind}|{band}"
    if not allres[key]:
        print(f"{label[kind]:<14}{bl[band]:<12}  (无结果)"); continue
    a, k, tr = agg(key)
    final[key] = {"acc_mean": float(a.mean()), "acc_std": float(a.std()),
                  "kappa_mean": float(k.mean()), "kappa_std": float(k.std()),
                  "train_acc_mean": float(tr.mean()), "n_seeds": len(a)}
    print(f"{label[kind]:<14}{bl[band]:<12}{a.mean()*100:5.1f}% ± {a.std()*100:4.1f}%   "
          f"{k.mean():+.3f} ± {k.std():.3f}   train={tr.mean():.3f}")
print(f"\n{'随机猜测':<14}{'':<12} 50.0%             0.000")

json.dump({"seeds": SEEDS, "configs": [f"{k}|{b}" for k, b in CONFIGS],
           "final_mean_std_over_seeds": final, "raw": allres},
          open(OUT, "w"), indent=2, ensure_ascii=False)
print(f"\n结果已存 {OUT}\nDONE")
