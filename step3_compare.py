"""第三步:LaBraM(线性探针 / 全量微调) vs EEGNet(端到端),cross-subject 公平对比。
统一数据/划分/预处理/评估。防 leakage:按被试切;三模型同一划分;早停只用验证被试;
测试集只碰一次;只用固定 ×1e6 缩放(不引入测试集统计量)。
"""
import os, json, time, warnings, copy
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

SEED = 42
SUBJECTS = list(range(1, 51))      # 用已缓存的 1-50
RUNS = [4, 8, 12]                  # 左/右手想象
SFREQ = 200                        # LaBraM 要求
N_TIMES = 1000                     # 5s @ 200Hz -> 5 个 patch
PATCH = 200
CACHE = os.path.expanduser("~/mne_data/labram_pretrained")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
print(f"设备={DEV}  | 目标: 三模型 cross-subject 公平对比\n")

# ============ 1. 统一数据管道(三模型共用一份) ============
print("[1] 载入 + 预处理(8-30Hz -> 200Hz -> 5s/1000点 -> µV)")
subj_data = {}
for s in SUBJECTS:
    try:
        fns = eegbci.load_data(subjects=[s], runs=RUNS)
        raw = concatenate_raws([read_raw_edf(f, preload=True, verbose="ERROR") for f in fns])
        eegbci.standardize(raw)                          # 规范通道名 -> 标准 10-05
        raw.set_montage("standard_1005", on_missing="ignore")
        raw.filter(8., 30., verbose="ERROR")             # 和 EEGNet baseline 同口径
        events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3), verbose="ERROR")
        picks = pick_types(raw.info, eeg=True, meg=False, stim=False, eog=False, exclude="bads")
        ep = Epochs(raw, events, dict(T1=2, T2=3), tmin=-1., tmax=4., picks=picks,
                    baseline=None, preload=True, verbose="ERROR")
        ep.resample(SFREQ, verbose="ERROR")
        X = ep.get_data()[:, :, :N_TIMES] * 1e6          # V -> µV,裁到 1000 点
        y = ep.events[:, -1] - 2                          # T1->0(左), T2->1(右)
        if X.shape[1] != 64 or X.shape[2] != N_TIMES:
            print(f"   subject {s}: shape 异常 {X.shape},跳过"); continue
        subj_data[s] = (X.astype("float32"), y.astype("int64"))
        if "CHS_INFO" not in globals():
            CHS_INFO = ep.info["chs"]                     # 给 LaBraM 用(含通道名+位置)
    except Exception as e:
        print(f"   subject {s}: 失败 {type(e).__name__},跳过")
clean = sorted(subj_data)
print(f"   干净被试 {len(clean)} 个,每人 trial 数示例 {[len(subj_data[s][1]) for s in clean[:5]]}...")

# ============ 2. 按被试切分(固定 seed=42,三模型完全相同) ============
rng = np.random.RandomState(SEED)
perm = rng.permutation(clean)
test_subj = list(perm[:10]); val_subj = list(perm[10:15]); train_subj = list(perm[15:])
print(f"\n[2] 被试划分(按被试,无 leakage): 训练 {len(train_subj)} / 验证 {len(val_subj)} / 测试 {len(test_subj)}")
print(f"   测试被试(模拟新用户): {sorted(test_subj)}")

def gather(subs):
    Xs = np.concatenate([subj_data[s][0] for s in subs], 0)
    ys = np.concatenate([subj_data[s][1] for s in subs], 0)
    return Xs, ys
X_tr, y_tr = gather(train_subj)
X_va, y_va = gather(val_subj)
test_dict = {s: subj_data[s] for s in test_subj}
print(f"   训练池 {X_tr.shape}  验证 {X_va.shape}")

# ============ 工具:训练(早停) + 按被试评估 ============
@torch.no_grad()
def predict(model, X, bs=32):
    """分批前向。LaBraM 注意力是 O(tokens^2),整批喂会 MPS OOM,必须分批。"""
    model.eval(); out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.tensor(X[i:i + bs]).to(DEV)).argmax(1).cpu().numpy())
    return np.concatenate(out)

def train(model, trainable, lr, wd, max_ep, patience, bs, tag):
    model.to(DEV)
    opt = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
    lossfn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=bs, shuffle=True)
    best_val, best_state, pc = -1, None, 0
    t0 = time.time()
    for ep in range(max_ep):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); loss = lossfn(model(xb.to(DEV)), yb.to(DEV)); loss.backward(); opt.step()
        va = float((predict(model, X_va) == y_va).mean())                 # 分批,防 OOM
        ta = float((predict(model, X_tr[:600]) == y_tr[:600]).mean())     # 抽样算 train acc
        if va > best_val:
            best_val, pc = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            pc += 1
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"   [{tag}] ep{ep+1:2d} train_acc={ta:.3f} val_acc={va:.3f} best_val={best_val:.3f} ({time.time()-t0:.0f}s)")
        if pc >= patience:
            print(f"   [{tag}] 早停 @ ep{ep+1}  best_val={best_val:.3f}"); break
    model.load_state_dict(best_state)
    return model, best_val

def eval_per_subject(model):
    model.eval(); rows = []
    for s, (Xs, ys) in test_dict.items():
        pred = predict(model, Xs)                 # 分批,防 OOM
        rows.append({"subject": int(s), "n": int(len(ys)),
                     "acc": float(accuracy_score(ys, pred)),
                     "kappa": float(cohen_kappa_score(ys, pred))})
    return rows

def summarize(rows):
    a = np.array([r["acc"] for r in rows]); k = np.array([r["kappa"] for r in rows])
    return dict(acc_mean=float(a.mean()), acc_std=float(a.std()),
               kappa_mean=float(k.mean()), kappa_std=float(k.std()))

def transplant_labram():
    """from_pretrained(3000) -> 移植到 n_times=1000(时间嵌入裁切)。返回一个全新模型。"""
    full = InterpolatedLaBraM.from_pretrained("braindecode/labram-pretrained", chs_info=CHS_INFO,
            n_times=3000, n_outputs=2, patch_size=PATCH, strict=False, cache_dir=CACHE)
    m = InterpolatedLaBraM(chs_info=CHS_INFO, n_times=N_TIMES, n_outputs=2, patch_size=PATCH)
    sf, sd = full.state_dict(), m.state_dict()
    new = {}
    for k, v in sd.items():
        if k in sf and sf[k].shape == v.shape: new[k] = sf[k]
        elif k == "temporal_embedding" and k in sf: new[k] = sf[k][:, :v.shape[1], :].clone()
        else: new[k] = v
    m.load_state_dict(new, strict=False)
    return m

results = {}

# ============ 模型 A: EEGNet 端到端 ============
print("\n[A] EEGNet 端到端")
torch.manual_seed(0)
eeg = EEGNet(n_chans=64, n_outputs=2, n_times=N_TIMES, drop_prob=0.5)
eeg, eeg_bv = train(eeg, eeg.parameters(), lr=1e-3, wd=1e-4, max_ep=60, patience=10, bs=32, tag="EEGNet")
results["EEGNet_e2e"] = {"rows": eval_per_subject(eeg), "best_val": eeg_bv}
eeg.to("cpu");  torch.mps.empty_cache() if DEV == "mps" else None   # 释放显存给下一个模型

# ============ 模型 B: LaBraM 线性探针(冻结骨干) ============
print("\n[B] LaBraM 线性探针(冻结骨干,只训头)")
torch.manual_seed(0)
probe = transplant_labram()
for n, p in probe.named_parameters():
    p.requires_grad = n.startswith("final_layer.")
head_params = [p for p in probe.parameters() if p.requires_grad]
print(f"   可训练参数(仅头)={sum(p.numel() for p in head_params)}")
probe, probe_bv = train(probe, head_params, lr=1e-3, wd=1e-4, max_ep=40, patience=8, bs=32, tag="探针")
results["LaBraM_probe"] = {"rows": eval_per_subject(probe), "best_val": probe_bv}
probe.to("cpu");  torch.mps.empty_cache() if DEV == "mps" else None   # 释放显存给全量微调

# ============ 模型 C: LaBraM 全量微调(放开骨干,小 lr 防过拟合) ============
print("\n[C] LaBraM 全量微调(放开骨干,lr=1e-4 + 早停 防过拟合)")
torch.manual_seed(0)
ft = transplant_labram()
for p in ft.parameters(): p.requires_grad = True
print(f"   可训练参数(全模型)={sum(p.numel() for p in ft.parameters())}")
ft, ft_bv = train(ft, ft.parameters(), lr=1e-4, wd=1e-4, max_ep=30, patience=8, bs=16, tag="全量FT")
results["LaBraM_fullFT"] = {"rows": eval_per_subject(ft), "best_val": ft_bv}

# ============ 汇总 + 存文件 ============
print("\n" + "=" * 64)
print("================ 三模型 cross-subject 对比 ================")
print(f"测试: {len(test_subj)} 个新被试,每人单独算 acc/kappa,报 mean±std\n")
order = [("EEGNet 端到端", "EEGNet_e2e"),
         ("LaBraM 线性探针", "LaBraM_probe"),
         ("LaBraM 全量微调", "LaBraM_fullFT")]
print(f"{'模型':<18}{'Accuracy':<20}{'Kappa':<20}")
print("-" * 58)
summary = {}
for name, key in order:
    s = summarize(results[key]["rows"]); summary[key] = s
    print(f"{name:<16}{s['acc_mean']*100:5.1f}% ± {s['acc_std']*100:4.1f}%     "
          f"{s['kappa_mean']:+.3f} ± {s['kappa_std']:.3f}")
print(f"\n{'随机猜测 baseline':<16}  50.0%                0.000")

# 每个测试被试明细
print("\n— 每个测试被试 accuracy 明细 —")
hdr = "subject  " + "  ".join(f"{n:<14}" for n, _ in order)
print(hdr)
for i, s in enumerate(sorted(test_subj)):
    line = f"  {s:<7}"
    for _, key in order:
        row = next(r for r in results[key]["rows"] if r["subject"] == s)
        line += f"  {row['acc']*100:5.1f}%        "
    print(line)

out = {"config": {"subjects": [int(x) for x in clean], "train": [int(x) for x in train_subj],
                  "val": [int(x) for x in val_subj], "test": [int(x) for x in test_subj],
                  "sfreq": SFREQ, "n_times": N_TIMES, "filter": "8-30Hz", "scale": "1e6_uV"},
       "results": results, "summary": summary}
with open(os.path.expanduser("~/bci-project/step3_results.json"), "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\n结果已存 ~/bci-project/step3_results.json")
print("DONE")
