"""Edge-deployment check on held-out Cho2017 subjects: does int8 quantization keep cross-subject accuracy?
Same split/training as step3d_matrix (broad 0.1-75 Hz band, 10 test subjects never seen in training).
For each trained model: fp32 PyTorch, fp32 ONNX Runtime, int8 dynamic (weights) and int8 static (QDQ, calibrated
on training subjects only). Reports accuracy, ONNX file size, single-thread CPU latency per 5-s window, FLOPs.
Usage: venv/bin/python heldout/edge_quant.py <EEGNet|fullFT|cbramodFT> <seed>
Results append to heldout/edge_quant_results.json; weights cached in heldout/edge/.
"""
import os, sys, json, time, pickle, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("MI_DATASET", "cho2017")
import numpy as np
import torch, torch.nn as nn
import sys as _sys, os as _os; _sys.path.insert(0, _os.path.expanduser("~/bci-project/heldout")); import mps_groupconv  # noqa  MPS grouped-conv bug fix (docs/mps_groupconv_bug.md)

ROOT = os.path.expanduser("~/bci-project")
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/heldout")
import step3d_matrix as S
import mps_sdpa  # noqa: F401
from torch.utils.data import TensorDataset, DataLoader

BAND, OUT, WDIR = "broad_0.1-75", f"{ROOT}/heldout/edge_quant_results.json", f"{ROOT}/heldout/edge"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def data(seed):
    meta = pickle.load(open(S.META, "rb"))
    D = pickle.load(open(S.band_cache(BAND), "rb"))
    D = {k: (X * np.float32(S.CHO_SCALE_DEFAULT), y) for k, (X, y) in D.items()}
    perm = np.random.RandomState(seed).permutation(meta["clean"])
    test_s, val_s, train_s = list(perm[:10]), list(perm[10:15]), list(perm[15:])
    cat = lambda subs: (np.concatenate([D[s][0] for s in subs]), np.concatenate([D[s][1] for s in subs]))
    return meta["chs_info"], D, test_s, cat(train_s), cat(val_s)


def build(kind, chs_info):
    from braindecode.models import EEGNet, InterpolatedLaBraM, CBraMod
    if kind == "EEGNet":
        return EEGNet(n_chans=64, n_outputs=2, n_times=S.N_TIMES, drop_prob=0.5), (1e-3, 60, 10, 32)
    if kind == "cbramodFT":
        m = CBraMod.from_pretrained("braindecode/cbramod-pretrained", n_outputs=2, n_chans=64,
                                    n_times=S.N_TIMES, cache_dir=S.CACHE)
        m.register_forward_pre_hook(lambda mod, inp: (inp[0] * 0.01,))   # CBraMod expects units of 100 uV
        return m, (1e-4, 30, 8, 16)
    full = InterpolatedLaBraM.from_pretrained("braindecode/labram-pretrained", chs_info=chs_info, n_times=3000,
                                              n_outputs=2, patch_size=S.PATCH, strict=False, cache_dir=S.CACHE)
    m = InterpolatedLaBraM(chs_info=chs_info, n_times=S.N_TIMES, n_outputs=2, patch_size=S.PATCH)
    sf, new = full.state_dict(), {}
    for k, v in m.state_dict().items():   # same transplant as step3d_matrix: keep shapes, crop temporal embedding
        if k in sf and sf[k].shape == v.shape: new[k] = sf[k]
        elif k == "temporal_embedding" and k in sf: new[k] = sf[k][:, :v.shape[1], :].clone()
        else: new[k] = v
    m.load_state_dict(new, strict=False)
    return m, (1e-4, 30, 8, 16)


@torch.no_grad()
def predict_torch(m, X, dev, bs=32):
    m.eval().to(dev)
    return np.concatenate([m(torch.tensor(X[i:i + bs]).to(dev)).argmax(1).cpu().numpy() for i in range(0, len(X), bs)])


def train(m, hp, tr, va):
    lr, max_ep, patience, bs = hp
    m.to(DEV); opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4); lf = nn.CrossEntropyLoss()
    ld = DataLoader(TensorDataset(torch.tensor(tr[0]), torch.tensor(tr[1])), batch_size=bs, shuffle=True)
    best, state, bad = -1, None, 0
    for ep in range(max_ep):
        m.train()
        for xb, yb in ld:
            opt.zero_grad(); lf(m(xb.to(DEV)), yb.to(DEV)).backward(); opt.step()
        acc = float((predict_torch(m, va[0], DEV) == va[1]).mean())
        if acc > best: best, bad, state = acc, 0, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else: bad += 1
        if bad >= patience: break
    m.load_state_dict(state); print(f"   trained to ep{ep + 1}, best val {best:.3f}", flush=True)
    return m


def per_subject_acc(pred_fn, D, test_s):
    return float(np.mean([(pred_fn(D[s][0]) == D[s][1]).mean() for s in test_s]))


def ort_session(path):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def ort_predict(sess, X, bs=32):
    name = sess.get_inputs()[0].name
    return np.concatenate([sess.run(None, {name: X[i:i + bs].astype(np.float32)})[0].argmax(1) for i in range(0, len(X), bs)])


def latency_ms(sess, x1, n=100):
    name = sess.get_inputs()[0].name
    for _ in range(10): sess.run(None, {name: x1})
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); sess.run(None, {name: x1}); t.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(t))


def main(kind, seed):
    seed = int(seed); key = f"{kind}|seed{seed}"
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if key in res: print(f"{key} done, skip"); return
    os.makedirs(WDIR, exist_ok=True); torch.manual_seed(seed)
    chs_info, D, test_s, tr, va = data(seed)
    m, hp = build(kind, chs_info)
    wpath = f"{WDIR}/{kind}_seed{seed}.pt"
    if os.path.exists(wpath): m.load_state_dict(torch.load(wpath))
    else: m = train(m, hp, tr, va); torch.save(m.state_dict(), wpath)
    m.cpu().eval()
    from torch.nn.utils import parametrize   # max-norm parametrizations call renorm (no ONNX op); bake them into the weights
    for mod in list(m.modules()):
        if parametrize.is_parametrized(mod, "weight"):
            parametrize.remove_parametrizations(mod, "weight", leave_parametrized=True)

    row = {"params": int(sum(p.numel() for p in m.parameters())),
           "acc_fp32_torch": per_subject_acc(lambda X: predict_torch(m, X, "cpu"), D, test_s)}
    x1 = va[0][:1].astype(np.float32)
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc: m(torch.tensor(x1))
    row["mflops_per_window"] = fc.get_total_flops() / 1e6

    from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, QuantFormat, CalibrationDataReader
    fp32 = f"{WDIR}/{kind}_seed{seed}_fp32.onnx"
    torch.onnx.export(m, torch.tensor(x1), fp32, input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "b"}, "logits": {0: "b"}}, opset_version=17, dynamo=False)

    class Calib(CalibrationDataReader):   # calibration windows come from TRAINING subjects only
        def __init__(self):
            idx = np.random.RandomState(0).choice(len(tr[0]), 200, replace=False)
            self.it = iter([{"x": tr[0][i:i + 1].astype(np.float32)} for i in idx])
        def get_next(self): return next(self.it, None)

    variants = {"fp32": fp32, "int8_dynamic": fp32.replace("fp32", "int8dyn"), "int8_static": fp32.replace("fp32", "int8qdq")}
    quantize_dynamic(fp32, variants["int8_dynamic"], weight_type=QuantType.QInt8)
    quantize_static(fp32, variants["int8_static"], Calib(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    for name, path in variants.items():
        sess = ort_session(path)
        row[f"acc_{name}"] = per_subject_acc(lambda X: ort_predict(sess, X), D, test_s)
        row[f"size_kb_{name}"] = os.path.getsize(path) / 1024
        row[f"latency_ms_{name}"] = latency_ms(sess, x1)
        print(f"   {name:13s} acc {row[f'acc_{name}'] * 100:5.1f}%  {row[f'size_kb_{name}']:9.0f} KB  "
              f"{row[f'latency_ms_{name}']:7.2f} ms", flush=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res[key] = row; json.dump(res, open(OUT, "w"), indent=2)
    print(f"{key}: {json.dumps(row)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
