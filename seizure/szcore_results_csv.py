"""Re-implement SzCORE's 4-evaluate/evaluate.py on our hypothesis folder and write its results.csv.

Mirrors esl-epfl/sz-validation-framework: 256 Hz masks built from the TSVs, default timescoring
SampleScoring / EventScoring parameters, counts summed over recordings. Output is the CSV the
benchmark form uploads.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
from timescoring import annotations, scoring
sys.path.insert(0, os.path.dirname(__file__))
import szcore_siena as z

FS = 256


def to_mask(tsv):
    df = pd.read_csv(tsv, sep="\t")
    m = np.zeros(int(df["recordingDuration"].iloc[0] * FS))
    for r in df.itertuples():
        if r.eventType != "bckg":
            m[round(r.onset * FS): round(r.onset + r.duration) * FS] = 1
    return m


def scores(tp, fp, ref, dur):
    sens = tp / ref; prec = tp / (tp + fp) if tp + fp else float("nan")
    return sens, prec, 2 * sens * prec / (sens + prec), fp / (dur / 3600 / 24)


rows = []
hyp_root = Path(z.ROOT) / "seizure/szcore_hypothesis"
for ref_tsv in sorted(Path(z.BIDS).glob("sub-*/**/*.tsv")):
    ref = annotations.Annotation(to_mask(ref_tsv), FS)
    hyp = annotations.Annotation(to_mask(hyp_root / ref_tsv.relative_to(z.BIDS)), FS)
    s, e = scoring.SampleScoring(ref, hyp), scoring.EventScoring(ref, hyp)
    rows.append({"dataset": "siena", "subject": ref_tsv.name.split("_")[0], "file": ref_tsv.name,
                 "duration": len(ref.mask) / ref.fs, "tp_sample": s.tp, "fp_sample": s.fp,
                 "refTrue_sample": s.refTrue, "tp_event": e.tp, "fp_event": e.fp, "refTrue_event": e.refTrue})
df = pd.DataFrame(rows)
out = Path(z.ROOT) / "seizure/szcore_results_siena.csv"
df.to_csv(out)
for kind in ("sample", "event"):
    se, pr, f1, fpr = scores(df[f"tp_{kind}"].sum(), df[f"fp_{kind}"].sum(), df[f"refTrue_{kind}"].sum(), df["duration"].sum())
    print(f"{kind:6s} sens={se:.3f} prec={pr:.3f} F1={f1:.3f} FP/24h={fpr:.1f}")
print("wrote", out, len(df), "recordings")
