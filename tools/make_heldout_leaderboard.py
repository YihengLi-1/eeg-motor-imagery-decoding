#!/usr/bin/env python3
"""Build docs/heldout_leaderboard.md from the held-out result files + the provenance table.
Every number is computed by us under one subject-independent protocol; every cell carries a
provenance badge taken from docs/pretraining_overlap_eeg_fm.json (what the checkpoint saw in pretraining).
"""
import json, datetime, os
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
phys = json.load(open(f"{R}/step3c_fullmatrix_results.json"))["final_mean_std_over_seeds"]
cho = json.load(open(f"{R}/heldout/results_cho2017_matrix.json"))["final_mean_std_over_seeds"]
import statistics, os as _os
_tm = f"{R}/heldout/results_cho2017_timemil.json"
if _os.path.exists(_tm):
    _d = json.load(open(_tm)); _d = _d.get("results", _d)
    for _k, _rows in _d.items():
        _a = [r["mean_acc"] for r in _rows]
        if len(_a) >= 2:
            cho[_k] = {"acc_mean": statistics.mean(_a), "acc_std": statistics.stdev(_a), "n_seeds": len(_a)}
prov = json.load(open(f"{R}/docs/pretraining_overlap_eeg_fm.json"))
canon = {m["name"]: m.get("pretrain_canon", {}) for m in prov["models"]}
BAND = "broad_0.1-75"
rows = [  # (display, key, provenance model name or None for from-scratch)
    ("EEGNet (from scratch)", "EEGNet", None),
    ("TimeMIL (from scratch)", "TimeMIL", None),
    ("LaBraM-base, linear probe", "probe", "LaBraM"),
    ("LaBraM-base, full fine-tune", "fullFT", "LaBraM"),
    ("CBraMod, linear probe", "cbramodProbe", "CBraMod"),
    ("CBraMod, full fine-tune", "cbramodFT", "CBraMod"),
]
pending = ["TimeMIL on PhysioNet MI", "BIOT (3 checkpoints)", "EEGPT", "CodeBrain", "BENDR", "REVE", "MIREPNet (PhysionetMI and Cho2017 are in its pretraining)"]
dsets = [("PhysioNet MI (109 subj., left/right hand)", "EEGMMIDB", phys), ("Cho2017 (52 subj., left/right hand)", "Cho2017", cho)]
def badge(model, key):
    if model is None: return "S"
    lab = canon.get(model, {}).get(key)
    if lab is None: return "H"
    return {"IN": "I", "PARTIAL": "P", "RELATED": "R", "HELD-OUT": "H"}.get(lab.split("*")[0], "?")
def cell(res, key, model, dkey):
    v = res.get(f"{key}|{BAND}")
    if not v: return "–"
    return f"{v['acc_mean']*100:.1f} ± {v['acc_std']*100:.1f} [{badge(model, dkey)}]"
out = []
out.append("# Held-out leaderboard for EEG foundation models (reference implementation)\n")
out.append(f"Updated {datetime.date.today()}. Maintainer: Yiheng Li. Companion to the [pretraining-overlap table](pretraining_overlap_eeg_fm.md) and to the `datasets:` model-card metadata being added to `braindecode/*` checkpoints (braindecode PR #1493).\n")
out.append("**What is different here.** Every number is computed by us, not self-reported, under one subject-independent protocol, and every cell carries a provenance badge saying whether that checkpoint's pretraining corpus contained the evaluation dataset. A score is only a cross-subject result when the badge is **[H]**.\n")
out.append("Badges: **[H]** held out (dataset not in the checkpoint's stated pretraining corpus) · **[I]** in corpus · **[P]** same collection (a TUH sub-corpus while pretraining used TUEG) · **[R]** related release (e.g. SEED-V vs SEED) · **[S]** trained from scratch, no pretraining · **[?]** corpus not itemised by the authors.\n")
out.append("## Motor imagery, binary, accuracy (mean ± SD over 4 subject-level splits)\n")
out.append("| Checkpoint / training | " + " | ".join(d[0] for d in dsets) + " |")
out.append("|---|" + "---|" * len(dsets))
for disp, key, model in rows:
    out.append(f"| {disp} | " + " | ".join(cell(res, key, model, dkey) for _, dkey, res in dsets) + " |")
out.append("\n**Protocol.** Test subjects are never seen in fine-tuning: 4 random subject-level splits (seeds 0–3); 0.1–75 Hz band, 4-s windows; Adam with early stopping on a validation set of held-out subjects; identical windows, splits and budget for every row. LaBraM and CBraMod start from the released braindecode weights; EEGNet and TimeMIL (InceptionTime backbone with time-aware MIL pooling, same splits and early stopping) are trained from scratch. Cho2017 amplitudes are rescaled by one global constant to PhysioNet's range for the pretrained models (EEGNet is scale-invariant). Code: `step3d_matrix.py`; raw results: `step3c_fullmatrix_results.json`, `heldout/results_cho2017_matrix.json`.\n")
out.append("**How to read it.** LaBraM's PhysioNet number is **[I]**: PhysioNet (EEGMMIDB) is listed in LaBraM's pretraining corpus, so 79.1% is not a cross-subject result for that checkpoint. Its Cho2017 number is **[H]**. CBraMod pretrains on TUEG only, so both of its numbers are **[H]**; it drops by a similar amount between the two datasets (82.3 → 71.4) as LaBraM (79.1 → 65.8), which is why most of the gap is the dataset, not the overlap (see the paper draft for the direct test).\n")
out.append("## Rows being added\n")
out.append("\n".join(f"- {p}" for p in pending) + "\n")
out.append("## Adding a checkpoint\n")
out.append("Open an issue with the Hub repo id. We run it under the protocol above and take the provenance badge from the checkpoint's model card `datasets:` field (or from the audit table when the card has none). Authors who disagree with a badge: the audit table records page-cited quotes; send the correction and we update both.\n")
open(f"{R}/docs/heldout_leaderboard.md", "w").write("\n".join(out))
print("\n".join(out))
