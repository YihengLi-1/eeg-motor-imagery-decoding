"""Per-subject sensitivity and false alarms for EEGNet, LaBraM and their ensemble (saved results only)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rp = json.load(open(f"{HERE}/results_repostproc.json"))
ens = json.load(open(f"{HERE}/results_ensemble.json"))


def table(ps):
    return {p["subject"]: p for p in (ps.values() if isinstance(ps, dict) else ps)}


M = {"EEGNet": table(rp["EEGNet"]["per_subject"]), "LaBraM": table(rp["fullFT"]["per_subject"]),
     "Ensemble": table(ens["per_subject"])}
subs = sorted(M["EEGNet"])
x = np.arange(len(subs)); w = 0.27
col = {"EEGNet": "#d62728", "LaBraM": "#1f77b4", "Ensemble": "#2ca02c"}

fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True)
for i, k in enumerate(M):
    sens = [M[k][s]["tp"] / M[k][s]["refTrue"] for s in subs]
    fa = [M[k][s]["fp"] * 24 / M[k][s]["hours"] for s in subs]
    a1.bar(x + (i - 1) * w, sens, w, color=col[k], label=k)
    a2.bar(x + (i - 1) * w, fa, w, color=col[k])
a1.set_ylabel("Sensitivity"); a1.set_ylim(0, 1.05); a1.legend(ncol=3, fontsize=8, frameon=False, loc="upper center",
                                                             bbox_to_anchor=(0.5, 1.28))
a2.set_ylabel("FA / 24 h")
labels = [f"{s.replace('sub-', 'P')}\n({M['EEGNet'][s]['refTrue']})" for s in subs]
a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=7)
a2.set_xlabel("Test subject (number of seizures)")
fr = subs.index("sub-10")
for a in (a1, a2):
    a.axvspan(fr - 0.45, fr + 0.45, color="0.9", zorder=0)
    a.tick_params(labelsize=7)
a1.text(fr, 1.0, "frontal", ha="center", va="top", fontsize=7)
fig.tight_layout()
out = f"{HERE}/figure_per_subject.png"
fig.savefig(out, dpi=300)
print(out)
