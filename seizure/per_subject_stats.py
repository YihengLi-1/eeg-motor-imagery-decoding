"""Per-subject breakdown and paired tests for the paper (reads saved results only; no retraining)."""
import json, os
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
rp = json.load(open(f"{HERE}/results_repostproc.json"))
ens = json.load(open(f"{HERE}/results_ensemble.json"))
M = {"EEGNet": rp["EEGNet"]["per_subject"], "LaBraM": rp["fullFT"]["per_subject"],
     "KD": rp["KD"]["per_subject"], "Ensemble": ens["per_subject"]}


def table(ps):
    ps = ps.values() if isinstance(ps, dict) else ps
    return {p["subject"]: p for p in ps}


T = {k: table(v) for k, v in M.items()}
subs = sorted(T["EEGNet"])
assert all(set(T[k]) == set(subs) for k in T), {k: len(T[k]) for k in T}


def metr(p):
    sens = p["tp"] / p["refTrue"] if p["refTrue"] else np.nan
    fa = p["fp"] * 24 / p["hours"]
    return sens, fa


rows = {}
print(f"{'subj':8}{'nSz':>4}{'hours':>7}  " + "  ".join(f"{k:>16}" for k in M))
for s in subs:
    r = {k: metr(T[k][s]) for k in M}
    rows[s] = r
    p = T["EEGNet"][s]
    print(f"{s:8}{p['refTrue']:>4}{p['hours']:>7.1f}  " +
          "  ".join(f"{T[k][s]['tp']:>2}/{T[k][s]['refTrue']:<2} FA{r[k][1]:>6.1f}" for k in M))

out = {}
for a, b in [("LaBraM", "EEGNet"), ("Ensemble", "EEGNet"), ("Ensemble", "LaBraM")]:
    sa = np.array([rows[s][a][0] for s in subs]); sb = np.array([rows[s][b][0] for s in subs])
    fa = np.array([rows[s][a][1] for s in subs]); fb = np.array([rows[s][b][1] for s in subs])
    ok = ~np.isnan(sa) & ~np.isnan(sb)
    ps = wilcoxon(sa[ok], sb[ok], zero_method="zsplit").pvalue
    pf = wilcoxon(fa, fb, zero_method="zsplit").pvalue
    out[f"{a}_vs_{b}"] = {"sens_median": [float(np.median(sa[ok])), float(np.median(sb[ok]))],
                          "fa_median": [float(np.median(fa)), float(np.median(fb))],
                          "p_sens": float(ps), "p_fa": float(pf), "n": int(ok.sum())}
    print(f"{a} vs {b}: median sens {np.median(sa[ok]):.2f} vs {np.median(sb[ok]):.2f} (p={ps:.3f}); "
          f"median FA/24h {np.median(fa):.1f} vs {np.median(fb):.1f} (p={pf:.3f}); n={ok.sum()}")
json.dump(out, open(f"{HERE}/results_per_subject_stats.json", "w"), indent=1)
