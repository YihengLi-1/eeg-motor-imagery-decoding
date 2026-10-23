"""Per-subject false alarms/24 h, LaBraM vs EEGNet, Siena (LOSO) and CHB-MIT (8-fold); saved results only."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def fa_pairs(tag):
    r = json.load(open(f"{HERE}/results_repostproc{tag}.json"))
    t = lambda ps: {p["subject"]: p for p in (ps.values() if isinstance(ps, dict) else ps)}
    a, b = t(r["EEGNet"]["per_subject"]), t(r["fullFT"]["per_subject"])
    fa = lambda p: p["fp"] * 24 / p["hours"]
    return np.array([[fa(a[s]), fa(b[s])] for s in sorted(a)])


fig, ax = plt.subplots(figsize=(3.4, 3.6))
for tag, lab, m, c in (("_loso", "Siena (in pretraining corpus)", "o", "#1f77b4"),
                       ("_chbmit", "CHB-MIT (outside corpus)", "^", "#d62728")):
    p = fa_pairs(tag)
    ax.scatter(p[:, 0] + 0.1, p[:, 1] + 0.1, marker=m, s=22, c=c, alpha=0.8, label=f"{lab}, n={len(p)}")
lim = (0.08, 400)
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("EEGNet false alarms / 24 h (+0.1)", fontsize=8)
ax.set_ylabel("LaBraM false alarms / 24 h (+0.1)", fontsize=8)
ax.tick_params(labelsize=7)
ax.text(0.12, 200, "LaBraM worse", fontsize=7); ax.text(20, 0.13, "LaBraM better", fontsize=7)
ax.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
fig.tight_layout()
out = f"{HERE}/figure_fa_scatter.png"
fig.savefig(out, dpi=300)
print(out)
