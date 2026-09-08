"""Render the README figure from committed split-A artifacts."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from summarize_results import load_results

rows = load_results()
fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
colors = ["#9baabb", "#9baabb", "#9baabb", "#397b9a", "#176457"]
for i, row in enumerate(rows):
    ax.barh(i, row["acc_mean"] * 100, xerr=row["acc_std"] * 100,
            color=colors[i], height=.58, capsize=4, error_kw={"elinewidth": 1.2})
    ax.text(102, i, f"{row['acc_mean']*100:.1f}% ± {row['acc_std']*100:.1f}%", va="center", fontsize=10)
ax.set_yticks(range(len(rows)), [row["model"] for row in rows])
ax.invert_yaxis()
ax.axvline(50, color="#59616a", linestyle="--", linewidth=1)
ax.set_xlim(0, 125)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Held-out accuracy (%) | dashed line: 50% chance level")
ax.set_title("Preprocessing sensitivity on one subject split", loc="left", fontsize=15, pad=18)
ax.spines[["top", "right"]].set_visible(False)
ax.text(0, -0.24, "10 held-out subjects; bars show mean ± population SD across subjects.\nSaved results, not a new training run. Full fine-tuning is not replicated across splits.",
        transform=ax.transAxes, fontsize=9, color="#505862", va="top")
output = Path(__file__).resolve().parents[1] / "docs" / "heldout-results.png"
output.parent.mkdir(exist_ok=True)
fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
print(output)
