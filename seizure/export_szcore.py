"""Export ensemble detections in the SzCORE hypothesis format and score them per recording.

For each fold, the validation-selected (weight, threshold, smoothing) from ensemble.py is applied to the
test patients' traces; the 1 Hz detection mask is cut back into the original recordings and written as
BIDS-style `*_events.tsv` files (SzCORE columns). Each recording is then scored against the reference
annotations with timescoring, as SzCORE's own `evaluate` package does, and the counts are summed.

Usage: python seizure/export_szcore.py   -> seizure/szcore_hypothesis/ (+ .zip) and a printed summary
"""
import os, sys, json, glob, shutil
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import szcore_siena as z
import repostproc as rp

HYP = f"{z.ROOT}/seizure/szcore_hypothesis"
COLS = ["onset", "duration", "eventType", "confidence", "channels", "dateTime", "recordingDuration"]


def mask_to_events(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def main():
    ens = json.load(open(f"{z.ROOT}/seizure/results_ensemble.json"))
    shutil.rmtree(HYP, ignore_errors=True)
    rows = []
    for f in ens["folds"]:
        fold, w, thr, k = int(f["fold"]), f["w_labram"], f["threshold"], f["smooth_s"]
        _, (ts, ta) = rp.traces(f"{z.MODELS}/fullFT_fold{fold}_traces.npz")
        _, (_, tb) = rp.traces(f"{z.MODELS}/EEGNet_fold{fold}_traces.npz")
        for s, (pa, mask), (pb, _) in zip(ts, ta, tb):
            det = z.to_events(w * pa + (1 - w) * pb, thr, k)
            bounds = np.load(f"{z.CACHE}/{s}.npz")["bounds"]
            refs = sorted(glob.glob(f"{z.BIDS}/{s}/**/*_events.tsv", recursive=True))
            assert len(refs) == len(bounds) - 1, (s, len(refs), bounds)
            for ref_tsv, a, b in zip(refs, bounds[:-1], bounds[1:]):
                ref = pd.read_csv(ref_tsv, sep="\t")
                dur, dt = float(ref["recordingDuration"].iloc[0]), ref["dateTime"].iloc[0]
                ev = [(on, off - on) for on, off in mask_to_events(det[a:b])]
                out = pd.DataFrame([{"onset": float(on), "duration": float(d), "eventType": "sz", "confidence": "n/a",
                                     "channels": "n/a", "dateTime": dt, "recordingDuration": dur} for on, d in ev]
                                   or [{"onset": 0.0, "duration": dur, "eventType": "bckg", "confidence": "n/a",
                                        "channels": "n/a", "dateTime": dt, "recordingDuration": dur}], columns=COLS)
                dst = os.path.join(HYP, os.path.relpath(ref_tsv, z.BIDS))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                out.to_csv(dst, sep="\t", index=False, float_format="%.2f")
                # score this recording on its own, from the two TSV files (reference vs. hypothesis)
                n = int(np.ceil(dur))
                to_mask = lambda df: np.isin(np.arange(n), np.concatenate(
                    [np.arange(int(r.onset), min(n, int(np.ceil(r.onset + r.duration))))
                     for r in df[df.eventType.astype(str).str.startswith("sz")].itertuples()] or [np.array([], int)]))
                r = z.score(to_mask(ref), to_mask(out)); r["subject"] = s; r["recording"] = os.path.basename(ref_tsv)
                rows.append(r)
    agg = z.aggregate(rows)
    print(f"per-recording SzCORE scoring: recordings={len(rows)} sens={agg['sensitivity']:.3f} "
          f"prec={agg['precision']:.3f} F1={agg['f1']:.3f} FP/24h={agg['fp_per_24h']:.1f} seizures={agg['n_seizures']}")
    json.dump({"pooled_per_recording": agg, "per_recording": rows},
              open(f"{z.ROOT}/seizure/results_ensemble_szcore_format.json", "w"), indent=2, default=str)
    shutil.make_archive(HYP, "zip", HYP)


if __name__ == "__main__":
    main()
