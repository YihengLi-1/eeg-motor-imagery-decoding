"""Distil each fold's fine-tuned LaBraM (5.8M params) into an EEGNet student (~2k params).

Question for implant-scale decoders: how much of the foundation model's cross-patient seizure
detection survives in a model ~2,600x smaller? Same folds, threshold selection and SzCORE scoring
as szcore_siena.py, so the three rows (EEGNet scratch / LaBraM / distilled EEGNet) compare directly.

Usage:  python seizure/distill.py run <fold>      (needs seizure/_models/fullFT_fold<k>.pt)
        python seizure/distill.py summary
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import szcore_siena as z

OUT = f"{z.ROOT}/seizure/results_distill.json"
# Variant "KDbg": the teacher's strength is rejecting background EEG, so give the student far more
# background windows (teacher-labelled) than the 1:5 sampling used for supervised training.
BG = int(os.environ.get("KD_BG_RATIO", z.BG_RATIO))
TAG = "KD" if BG == z.BG_RATIO else f"KDbg{BG}"
T, ALPHA = 2.0, 0.5            # softmax temperature, weight of the hard-label loss


def load_results():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def cpu_ms_per_window(m):
    import torch
    m = m.to("cpu").eval(); x = torch.randn(64, len(z.CHANNELS), z.N_TIMES)
    with torch.no_grad():
        m(x[:8]); t0 = time.perf_counter()
        for _ in range(3): m(x)
    return (time.perf_counter() - t0) / (3 * 64) * 1000


def do_run(fold):
    import torch, torch.nn as nn, torch.nn.functional as Fn
    fold = int(fold); key = f"{TAG}|fold{fold}"
    z.BG_RATIO = BG
    if key in load_results():
        print(f"   [{key}] done, skip"); return
    DEV = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(fold); rng = np.random.RandomState(fold)
    F = z.folds()[fold]

    teacher = z.build_model("fullFT")
    teacher.load_state_dict(torch.load(f"{z.MODELS}/fullFT_fold{fold}.pt"))
    teacher.to(DEV).eval()

    Xtr, ytr = z.sample_train(F["train"], rng)
    with torch.no_grad():
        soft = torch.cat([torch.softmax(teacher(torch.tensor(Xtr[i:i + 64]).to(DEV)) / T, 1).cpu()
                          for i in range(0, len(Xtr), 64)])
    print(f"   train windows {len(ytr)} (seizure {ytr.sum()}), teacher soft labels ready")

    student = z.build_model("EEGNet").to(DEV)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()                  # same as szcore_siena v2: 1:5 sampling, no extra class weights
    best, best_state, patience, t0 = -1, None, 0, time.time()
    for ep in range(40):
        student.train(); idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 64):
            j = idx[i:i + 64]
            logits = student(torch.tensor(Xtr[j]).to(DEV))
            kd = Fn.kl_div(Fn.log_softmax(logits / T, 1), soft[j].to(DEV), reduction="batchmean") * T * T
            loss = ALPHA * ce(logits, torch.tensor(ytr[j]).to(DEV)) + (1 - ALPHA) * kd
            opt.zero_grad(); loss.backward(); opt.step()
        f1 = z.best_postproc([z.subject_prob_trace(student, s, DEV) for s in F["val"]])[2]
        print(f"      ep{ep+1} val_F1={f1:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if best_state is None or f1 > best:
            best, patience = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        else:
            patience += 1
            if patience >= 6: break
    student.load_state_dict(best_state)
    thr, k, _ = z.best_postproc([z.subject_prob_trace(student, s, DEV) for s in F["val"]])
    rows, traces = [], {}
    for s in F["val"]:
        p, mask = z.subject_prob_trace(student, s, DEV)
        traces[f"val_{s}_prob"], traces[f"val_{s}_mask"] = p.astype("float32"), mask
    for s in F["test"]:
        p, mask = z.subject_prob_trace(student, s, DEV)
        traces[f"test_{s}_prob"], traces[f"test_{s}_mask"] = p.astype("float32"), mask
        r = z.score(mask, z.to_events(p, thr, k)); r["subject"] = s; rows.append(r)
    np.savez_compressed(f"{z.MODELS}/{TAG}_fold{fold}_traces.npz", **traces)

    res = load_results()
    res[key] = {"fold": fold, "test": F["test"], "val": F["val"], "threshold": thr, "smooth_s": k,
                "best_val_f1": best,
                "per_subject": rows, **z.aggregate(rows),
                "student_params": sum(p.numel() for p in student.parameters()),
                "teacher_params": sum(p.numel() for p in teacher.parameters()),
                "student_cpu_ms_per_window": cpu_ms_per_window(student),
                "teacher_cpu_ms_per_window": cpu_ms_per_window(teacher)}
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"   [{key}] F1={res[key]['f1']:.3f} sens={res[key]['sensitivity']:.3f} "
          f"FP/24h={res[key]['fp_per_24h']:.1f} ✓")


def do_summary():
    res = load_results()
    rows = [r for k, v in res.items() if k.startswith("KD|") for r in v["per_subject"]]
    if rows:
        s = z.aggregate(rows); s["n_folds"] = sum(k.startswith("KD|") for k in res)
        v = next(v for k, v in res.items() if k.startswith("KD|"))
        for k in ("student_params", "teacher_params", "student_cpu_ms_per_window", "teacher_cpu_ms_per_window"):
            s[k] = v[k]
        res["summary"] = s
        json.dump(res, open(OUT, "w"), indent=2)
        print(f"KD-EEGNet folds={s['n_folds']} sens={s['sensitivity']:.3f} prec={s['precision']:.3f} "
              f"F1={s['f1']:.3f} FP/24h={s['fp_per_24h']:.1f} | params {s['student_params']} vs "
              f"{s['teacher_params']} | CPU ms/window {s['student_cpu_ms_per_window']:.2f} vs "
              f"{s['teacher_cpu_ms_per_window']:.2f}")


if __name__ == "__main__":
    {"run": lambda: do_run(sys.argv[2]), "summary": do_summary}[sys.argv[1]]()
