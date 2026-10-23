"""Re-select post-processing from saved probability traces with a wider threshold grid.

v2 used THRESHOLDS starting at 0.3 and EEGNet picked 0.3 in every fold (grid edge). This script
re-runs the selection on the *validation* traces only, with thresholds down to 0.05, and scores
the test traces once. No retraining and no test information is used for selection.

Usage: python seizure/repostproc.py [EEGNet fullFT KD]
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import szcore_siena as z

z.THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
OUT = f"{z.ROOT}/seizure/results_repostproc{z.SUFFIX}.json"


def traces(path):
    d = np.load(path)
    def pairs(prefix):
        subs = sorted({k.split("_")[1] for k in d.files if k.startswith(prefix)})
        return subs, [(d[f"{prefix}_{s}_prob"], d[f"{prefix}_{s}_mask"].astype(bool)) for s in subs]
    return pairs("val"), pairs("test")


def main(kinds):
    res = {}
    for kind in kinds:
        rows, folds = [], []
        for path in sorted(glob.glob(f"{z.MODELS}/{kind}_fold*_traces.npz")):
            (_, val), (tsubs, test) = traces(path)
            thr, k, vf1 = z.best_postproc(val)
            fr = []
            for s, (p, m) in zip(tsubs, test):
                r = z.score(m, z.to_events(p, thr, k)); r["subject"] = s; fr.append(r)
            rows += fr
            folds.append({"fold": path.split("fold")[1].split("_")[0], "threshold": thr, "smooth_s": k,
                          "val_f1": vf1, **z.aggregate(fr)})
        if not folds:
            continue
        res[kind] = {"pooled": z.aggregate(rows), "mean_fold_f1": float(np.mean([f["f1"] for f in folds])),
                     "std_fold_f1": float(np.std([f["f1"] for f in folds])), "folds": folds, "per_subject": rows}
        p = res[kind]["pooled"]
        print(f"{kind:7s} folds={len(folds)} sens={p['sensitivity']:.3f} prec={p['precision']:.3f} "
              f"F1pooled={p['f1']:.3f} F1/fold={res[kind]['mean_fold_f1']:.3f}±{res[kind]['std_fold_f1']:.3f} "
              f"FP/24h={p['fp_per_24h']:.1f} thr={[f['threshold'] for f in folds]}")
    json.dump(res, open(OUT, "w"), indent=2, default=str)


if __name__ == "__main__":
    main(sys.argv[1:] or ["EEGNet", "fullFT"])
