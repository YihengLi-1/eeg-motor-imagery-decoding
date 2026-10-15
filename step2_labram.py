"""第二步:把基础模型换成 LaBraM,用教程那份 BNCI 数据 + MPS 验证加载与微调。
目标:验证 LaBraM 能在 M3 上用 from_pretrained 加载 + 微调跑通。不碰用户自己的数据。
"""
import os, time, warnings
warnings.simplefilter("ignore")
import mne; mne.set_log_level("ERROR")
import numpy as np
import torch

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.models import InterpolatedLaBraM
from braindecode import EEGClassifier

torch.manual_seed(12); np.random.seed(12)

N_TIMES = 800      # 4s @ 200Hz = 4 个 patch(patch_size=200)
PATCH = 200
SUBJECT = 3
CLASSES = ["feet", "left_hand", "right_hand"]
CACHE = os.path.expanduser("~/mne_data/labram_pretrained")

print(f"[1/5] 载入 BNCI subject {SUBJECT},重采样到 200Hz")
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[SUBJECT])
montage = mne.channels.make_standard_montage("standard_1020")
for ds in dataset.datasets:
    ds.raw.pick_types(eeg=True)
    ds.raw.set_montage(montage)
    ds.raw.resample(200)
chs_info = dataset.datasets[0].raw.info["chs"]
ch_names = [c["ch_name"] for c in chs_info]
print(f"   通道数={len(ch_names)}  采样率={dataset.datasets[0].raw.info['sfreq']}")

print(f"[2/5] 切窗 window={N_TIMES}(=4s,1 窗/trial)")
mapping = {c: i for i, c in enumerate(CLASSES)}
windows = create_windows_from_events(
    dataset, preload=True, mapping=mapping,
    window_size_samples=N_TIMES, window_stride_samples=N_TIMES,
)
meta = windows.get_metadata()
n_win = len(meta)
print(f"   窗口数={n_win}  类别分布={meta['target'].value_counts().to_dict()}")
assert n_win > 0, "切窗得到 0 个窗口,需要调整 window 参数"

print("[3/5] from_pretrained(n_times=3000) 载入 LaBraM-Base 预训练权重")
t = time.time()
full = InterpolatedLaBraM.from_pretrained(
    "braindecode/labram-pretrained", chs_info=chs_info,
    n_times=3000, n_outputs=len(CLASSES), patch_size=PATCH,
    strict=False, cache_dir=CACHE,
)
print(f"   OK  {time.time()-t:.1f}s  参数={sum(p.numel() for p in full.parameters())/1e6:.2f}M")

print(f"[4/5] 移植权重到 n_times={N_TIMES} 的模型(时间嵌入按 patch 数裁切)")
model = InterpolatedLaBraM(
    chs_info=chs_info, n_times=N_TIMES, n_outputs=len(CLASSES), patch_size=PATCH,
)
sd_full, sd = full.state_dict(), model.state_dict()
new, n_xfer, n_slice, n_keep = {}, 0, 0, 0
for k, v in sd.items():
    if k in sd_full and sd_full[k].shape == v.shape:
        new[k] = sd_full[k]; n_xfer += 1
    elif k == "temporal_embedding" and k in sd_full:
        new[k] = sd_full[k][:, : v.shape[1], :].clone(); n_slice += 1
    else:
        new[k] = v; n_keep += 1
model.load_state_dict(new, strict=False)
print(f"   迁移张量={n_xfer}  裁切 temporal_embedding={n_slice}  保留新初始化={n_keep}(头/插值层)")

# 冻结骨干,只训练分类头(和第一步一致:线性探针)
for name, p in model.named_parameters():
    p.requires_grad = name.startswith("final_layer.")
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   可训练参数(仅分类头)={n_trainable}")

print("[5/5] 在 MPS 上微调 10 轮")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"   设备={dev}")
clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    optimizer__lr=0.005,
    batch_size=16,
    callbacks=["accuracy"],
    classes=range(len(CLASSES)),
    device=dev,
)
t = time.time()
clf.fit(windows, y=meta["target"], epochs=10)
print(f"   训练完成 总耗时={time.time()-t:.1f}s")
print("DONE")
