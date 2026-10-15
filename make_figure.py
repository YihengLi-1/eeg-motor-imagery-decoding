"""可重复运行的作图脚本。读取多 seed 矩阵结果,画一张分组柱状图。
现在跑能出当前进度版;所有 seed 跑完再跑一次即最终版。
用法: ./venv/bin/python make_figure.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/bci-project")
RES = f"{ROOT}/step3c_fullmatrix_results.json"
OUT = f"{ROOT}/figure_band_x_model.png"

res = (json.load(open(RES)).get("results")) or {}

MODELS = [("EEGNet", "EEGNet\n(from scratch)"),
          ("probe", "LaBraM\n(linear probe)"),
          ("fullFT", "LaBraM\n(full fine-tune)")]
BANDS = [("narrow_8-30", "Narrow 8-30 Hz", "#9aa0a6"),
         ("broad_0.1-75", "Broad 0.1-75 Hz", "#1d9e75")]

def stats(kind, band):
    rows = res.get(f"{kind}|{band}", [])
    if not rows:
        return None, None, 0
    a = np.array([r["mean_acc"] * 100 for r in rows])
    return a.mean(), (a.std() if len(a) > 1 else 0.0), len(a)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(MODELS)); w = 0.38

for j, (bkey, blabel, color) in enumerate(BANDS):
    means, errs, ns, hatches = [], [], [], []
    for kind, _ in MODELS:
        m, s, n = stats(kind, bkey)
        means.append(m if m is not None else 0)
        errs.append(s if s is not None else 0)
        ns.append(n)
        hatches.append("" if n > 0 else "//")
    bars = ax.bar(x + (j - 0.5) * w, means, w, yerr=errs, capsize=4,
                  label=blabel, color=color, edgecolor="black", linewidth=0.6)
    for b, m, n, h in zip(bars, means, ns, hatches):
        if n == 0:
            b.set_facecolor("none"); b.set_hatch("//"); b.set_edgecolor("#bbb")
            ax.text(b.get_x() + b.get_width() / 2, 51, "pending", ha="center",
                    va="bottom", fontsize=8, color="#999", rotation=90)
        else:
            ax.text(b.get_x() + b.get_width() / 2, m + (errs[list(bars).index(b)] if False else 0) + 0.8,
                    f"{m:.0f}%\n(n={n})", ha="center", va="bottom", fontsize=9)

ax.axhline(50, ls="--", lw=1, color="#c0392b")
ax.text(len(MODELS) - 0.5, 50.4, "chance (50%)", color="#c0392b", fontsize=8, ha="right")
ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in MODELS])
ax.set_ylabel("Cross-subject accuracy (%)")
ax.set_ylim(45, 90)
ax.set_title("Cross-subject motor imagery: pretraining + matched band is what wins",
             fontsize=11)
ax.legend(title="Preprocessing band", loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"saved -> {OUT}")
for kind, _ in MODELS:
    for bkey, blab, _ in BANDS:
        m, s, n = stats(kind, bkey)
        print(f"  {kind:7} {bkey:13} " + (f"{m:.1f}% ± {s:.1f} (n={n})" if n else "pending"))
