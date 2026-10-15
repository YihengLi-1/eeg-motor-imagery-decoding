"""Sensitivity vs. false alarms per 24 h for every seizure-detection variant (pooled over test patients)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = os.path.dirname(os.path.abspath(__file__))
rp = json.load(open(f"{D}/results_repostproc.json")); en = json.load(open(f"{D}/results_ensemble.json"))
pts = [("EEGNet (2.2k params)", rp["EEGNet"]["pooled"], "#888888", "o"),
       ("LaBraM fine-tuned (5.8M)", rp["fullFT"]["pooled"], "#1b9e77", "s"),
       ("EEGNet distilled", rp["KD"]["pooled"], "#d95f02", "v"),
       ("EEGNet distilled, 30x background", rp["KDbg30"]["pooled"], "#d95f02", "^"),
       ("LaBraM + EEGNet average", en["pooled"], "#7570b3", "*")]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for name, p, c, m in pts:
    ax.scatter(p["fp_per_24h"], p["sensitivity"], s=170 if m == "*" else 80, c=c, marker=m, edgecolor="k", zorder=3)
    ax.annotate(f"{name}\nF1={p["f1"] + 1e-9:.2f}", (p["fp_per_24h"], p["sensitivity"]), textcoords="offset points",
                xytext=(8, -4), fontsize=8)
ax.set_xlabel("False alarms per 24 h (lower is better)"); ax.set_ylabel("Seizure sensitivity (higher is better)")
ax.set_xlim(0, 60); ax.set_ylim(0, 0.8); ax.grid(alpha=0.3)
ax.set_title("Patient-independent seizure detection, BIDS Siena, SzCORE event scoring", fontsize=9.5)
fig.tight_layout(); fig.savefig(f"{D}/figure_seizure_tradeoff.png", dpi=200)
print("saved")
