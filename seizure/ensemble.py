"""Average LaBraM and EEGNet seizure probabilities (complementary errors: few false alarms vs. high sensitivity).

Uses the saved per-fold traces; the mixing weight, threshold and smoothing are all chosen on the
validation patients of each fold, then the test patients are scored once.
Usage: python seizure/ensemble.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import szcore_siena as z
import repostproc as rp           # wider threshold grid + trace loader

WEIGHTS = (0.25, 0.5, 0.75)       # weight on LaBraM
OUT = f"{z.ROOT}/seizure/results_ensemble{z.SUFFIX}.json"


def main():
    rows, folds = [], []
    for fold in range(z.N_FOLDS):
        (vs, va), (ts, ta) = rp.traces(f"{z.MODELS}/fullFT_fold{fold}_traces.npz")
        (vs2, vb), (ts2, tb) = rp.traces(f"{z.MODELS}/EEGNet_fold{fold}_traces.npz")
        assert vs == vs2 and ts == ts2
        mix = lambda a, b, w: [(w * pa + (1 - w) * pb, ma) for (pa, ma), (pb, _) in zip(a, b)]
        best = max(((w,) + z.best_postproc(mix(va, vb, w)) for w in WEIGHTS), key=lambda r: r[3])
        w, thr, k, vf1 = best
        fr = []
        for s, (p, m) in zip(ts, mix(ta, tb, w)):
            r = z.score(m, z.to_events(p, thr, k)); r["subject"] = s; fr.append(r)
        rows += fr
        folds.append({"fold": fold, "w_labram": w, "threshold": thr, "smooth_s": k, "val_f1": vf1, **z.aggregate(fr)})
    res = {"pooled": z.aggregate(rows), "mean_fold_f1": float(np.mean([f["f1"] for f in folds])),
           "std_fold_f1": float(np.std([f["f1"] for f in folds])), "folds": folds, "per_subject": rows}
    json.dump(res, open(OUT, "w"), indent=2, default=str)
    p = res["pooled"]
    print(f"ENS     sens={p['sensitivity']:.3f} prec={p['precision']:.3f} F1pooled={p['f1']:.3f} "
          f"F1/fold={res['mean_fold_f1']:.3f}±{res['std_fold_f1']:.3f} FP/24h={p['fp_per_24h']:.1f} "
          f"w={[f['w_labram'] for f in folds]}")


if __name__ == "__main__":
    main()
