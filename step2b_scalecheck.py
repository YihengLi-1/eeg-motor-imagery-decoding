"""诊断:第二步准确率钉在 chance,验证是不是 LaBraM 输入幅度(µV)问题。
和 step2 一样,只多做一件事:把数据从伏特乘 1e6 转成 µV,再看准确率动不动。
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
N_TIMES, PATCH, SUBJECT = 800, 200, 3
CLASSES = ["feet", "left_hand", "right_hand"]
CACHE = os.path.expanduser("~/mne_data/labram_pretrained")
SCALE = float(os.environ.get("SCALE", "1e6"))  # 伏特 -> µV

dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[SUBJECT])
montage = mne.channels.make_standard_montage("standard_1020")
for ds in dataset.datasets:
    ds.raw.pick_types(eeg=True); ds.raw.set_montage(montage); ds.raw.resample(200)

# 量纲检查 + 缩放
std_before = np.std(dataset.datasets[0].raw.get_data())
for ds in dataset.datasets:
    ds.raw.apply_function(lambda x: x * SCALE)
std_after = np.std(dataset.datasets[0].raw.get_data())
print(f"数据标准差: 缩放前={std_before:.3e}  ×{SCALE:.0e} 后={std_after:.3e}")

mapping = {c: i for i, c in enumerate(CLASSES)}
windows = create_windows_from_events(dataset, preload=True, mapping=mapping,
                                     window_size_samples=N_TIMES, window_stride_samples=N_TIMES)
meta = windows.get_metadata()

full = InterpolatedLaBraM.from_pretrained("braindecode/labram-pretrained", chs_info=dataset.datasets[0].raw.info["chs"],
                                          n_times=3000, n_outputs=3, patch_size=PATCH, strict=False, cache_dir=CACHE)
model = InterpolatedLaBraM(chs_info=dataset.datasets[0].raw.info["chs"], n_times=N_TIMES, n_outputs=3, patch_size=PATCH)
sd_full, sd = full.state_dict(), model.state_dict()
new = {}
for k, v in sd.items():
    if k in sd_full and sd_full[k].shape == v.shape: new[k] = sd_full[k]
    elif k == "temporal_embedding" and k in sd_full: new[k] = sd_full[k][:, :v.shape[1], :].clone()
    else: new[k] = v
model.load_state_dict(new, strict=False)
for name, p in model.named_parameters():
    p.requires_grad = name.startswith("final_layer.")

dev = "mps" if torch.backends.mps.is_available() else "cpu"
clf = EEGClassifier(model, criterion=torch.nn.CrossEntropyLoss, optimizer=torch.optim.AdamW,
                    optimizer__lr=0.005, batch_size=16, callbacks=["accuracy"],
                    classes=range(3), device=dev)
print(f"设备={dev}  缩放={SCALE:.0e}  训练 5 轮 ...")
clf.fit(windows, y=meta["target"], epochs=5)
print("DONE")
