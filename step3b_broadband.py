"""第三步补充:给 LaBraM 换成预训练时的宽带 0.1-75Hz,重跑 探针 + 全量微调。
其他全不变(同被试、同 seed=42 划分、防 leakage 四条、同评估)。唯一变量=滤波频段。
EEGNet(8-30Hz)与窄带 LaBraM 是确定性结果,复用 step3,不在此重算。
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import mne; mne.set_log_level("ERROR")
from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws
from mne import Epochs, pick_types, events_from_annotations
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score
from braindecode.models import InterpolatedLaBraM

SEED, SUBJECTS, RUNS = 42, list(range(1, 51)), [4, 8, 12]
SFREQ, N_TIMES, PATCH = 200, 1000, 200
FILTER = (0.1, 75.0)            # ★ 唯一改动:LaBraM 预训练的宽带
CACHE = os.path.expanduser("~/mne_data/labram_pretrained")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
print(f"设备={DEV} | LaBraM 宽带 {FILTER[0]}-{FILTER[1]}Hz 敏感度对照\n")

# ---- 1. 数据(同 step3,只改滤波) ----
print(f"[1] 载入 + 预处理({FILTER[0]}-{FILTER[1]}Hz -> 200Hz -> 5s/1000点 -> µV)")
subj_data = {}
for s in SUBJECTS:
    try:
        fns = eegbci.load_data(subjects=[s], runs=RUNS)
        raw = concatenate_raws([read_raw_edf(f, preload=True, verbose="ERROR") for f in fns])
        eegbci.standardize(raw); raw.set_montage("standard_1005", on_missing="ignore")
        raw.filter(FILTER[0], FILTER[1], verbose="ERROR")
        events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3), verbose="ERROR")
        picks = pick_types(raw.info, eeg=True, meg=False, stim=False, eog=False, exclude="bads")
        ep = Epochs(raw, events, dict(T1=2, T2=3), tmin=-1., tmax=4., picks=picks,
                    baseline=None, preload=True, verbose="ERROR")
        ep.resample(SFREQ, verbose="ERROR")
        X = ep.get_data()[:, :, :N_TIMES] * 1e6
        y = ep.events[:, -1] - 2
        if X.shape[1] != 64 or X.shape[2] != N_TIMES: continue
        subj_data[s] = (X.astype("float32"), y.astype("int64"))
        if "CHS_INFO" not in globals(): CHS_INFO = ep.info["chs"]
    except Exception:
        continue
clean = sorted(subj_data)
print(f"   干净被试 {len(clean)} 个")

# ---- 2. 同 step3 的划分(seed 42 -> 完全相同的 35/5/10) ----
rng = np.random.RandomState(SEED); perm = rng.permutation(clean)
test_subj, val_subj, train_subj = list(perm[:10]), list(perm[10:15]), list(perm[15:])
print(f"[2] 划分 训练{len(train_subj)}/验证{len(val_subj)}/测试{len(test_subj)} | 测试被试 {sorted(int(s) for s in test_subj)}")
def gather(subs):
    return (np.concatenate([subj_data[s][0] for s in subs], 0),
            np.concatenate([subj_data[s][1] for s in subs], 0))
X_tr, y_tr = gather(train_subj); X_va, y_va = gather(val_subj)
test_dict = {s: subj_data[s] for s in test_subj}

# ---- 工具(同 step3:分批前向防 OOM + 早停 + 按被试评估) ----
@torch.no_grad()
def predict(model, X, bs=32):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.tensor(X[i:i+bs]).to(DEV)).argmax(1).cpu().numpy())
    return np.concatenate(out)

def train(model, trainable, lr, wd, max_ep, patience, bs, tag):
    model.to(DEV)
    opt = torch.optim.Adam(trainable, lr=lr, weight_decay=wd); lossfn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=bs, shuffle=True)
    best_val, best_state, pc, t0 = -1, None, 0, time.time()
    for ep in range(max_ep):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); nn.CrossEntropyLoss()(model(xb.to(DEV)), yb.to(DEV)).backward(); opt.step()
        va = float((predict(model, X_va) == y_va).mean())
        ta = float((predict(model, X_tr[:600]) == y_tr[:600]).mean())
        if va > best_val:
            best_val, pc = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: pc += 1
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"   [{tag}] ep{ep+1:2d} train_acc={ta:.3f} val_acc={va:.3f} best_val={best_val:.3f} ({time.time()-t0:.0f}s)")
        if pc >= patience:
            print(f"   [{tag}] 早停 @ ep{ep+1} best_val={best_val:.3f}"); break
    model.load_state_dict(best_state); return model

def eval_per_subject(model):
    rows = []
    for s, (Xs, ys) in test_dict.items():
        pred = predict(model, Xs)
        rows.append({"subject": int(s), "acc": float(accuracy_score(ys, pred)),
                     "kappa": float(cohen_kappa_score(ys, pred))})
    return rows

def summ(rows):
    a = np.array([r["acc"] for r in rows]); k = np.array([r["kappa"] for r in rows])
    return dict(acc_mean=float(a.mean()), acc_std=float(a.std()),
               kappa_mean=float(k.mean()), kappa_std=float(k.std()))

def transplant():
    full = InterpolatedLaBraM.from_pretrained("braindecode/labram-pretrained", chs_info=CHS_INFO,
            n_times=3000, n_outputs=2, patch_size=PATCH, strict=False, cache_dir=CACHE)
    m = InterpolatedLaBraM(chs_info=CHS_INFO, n_times=N_TIMES, n_outputs=2, patch_size=PATCH)
    sf, sd, new = full.state_dict(), m.state_dict(), {}
    for k, v in sd.items():
        if k in sf and sf[k].shape == v.shape: new[k] = sf[k]
        elif k == "temporal_embedding" and k in sf: new[k] = sf[k][:, :v.shape[1], :].clone()
        else: new[k] = v
    m.load_state_dict(new, strict=False); return m

results = {}

print("\n[B宽带] LaBraM 线性探针(冻结骨干)")
torch.manual_seed(0)
probe = transplant()
for n, p in probe.named_parameters(): p.requires_grad = n.startswith("final_layer.")
probe = train(probe, [p for p in probe.parameters() if p.requires_grad], 1e-3, 1e-4, 40, 8, 32, "宽带探针")
results["LaBraM_probe_broadband"] = eval_per_subject(probe)
probe.to("cpu"); torch.mps.empty_cache() if DEV == "mps" else None

print("\n[C宽带] LaBraM 全量微调(lr=1e-4 + 早停,盯过拟合)")
torch.manual_seed(0)
ft = transplant()
for p in ft.parameters(): p.requires_grad = True
ft = train(ft, ft.parameters(), 1e-4, 1e-4, 30, 8, 16, "宽带全量FT")
results["LaBraM_fullFT_broadband"] = eval_per_subject(ft)

print("\n" + "=" * 60)
print("===== LaBraM 宽带(0.1-75Hz)结果 =====")
for key, rows in results.items():
    s = summ(rows)
    print(f"{key:<26} acc {s['acc_mean']*100:5.1f}% ± {s['acc_std']*100:4.1f}%   kappa {s['kappa_mean']:+.3f} ± {s['kappa_std']:.3f}")

out = {"filter": "0.1-75Hz_broadband", "test_subjects": [int(s) for s in test_subj],
       "results": results, "summary": {k: summ(v) for k, v in results.items()}}
json.dump(out, open(os.path.expanduser("~/bci-project/step3b_broadband_results.json"), "w"), indent=2, ensure_ascii=False)
print("\n结果已存 ~/bci-project/step3b_broadband_results.json\nDONE")
