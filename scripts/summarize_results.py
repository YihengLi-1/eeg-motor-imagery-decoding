"""Inspect committed split-A results without training or third-party packages."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]


def load_results(root=ROOT):
    narrow = json.loads((root / "step3_results.json").read_text())
    broad = json.loads((root / "step3b_broadband_results.json").read_text())
    expected = set(narrow["config"]["test_subjects"])
    if len(expected) != 10 or set(broad["test_subjects"]) != expected:
        raise ValueError("Split-A artifacts must use the same ten held-out subjects")
    rows = []
    names = {
        "EEGNet_end_to_end": "EEGNet / 8-30 Hz",
        "LaBraM_linear_probe": "LaBraM probe / 8-30 Hz",
        "LaBraM_full_finetune": "LaBraM full fine-tune / 8-30 Hz",
    }
    for key, label in names.items():
        saved = narrow["summary_mean_std_over_10_test_subjects"][key]
        rows.append({"model": label, **saved})
    for key, label in (
        ("LaBraM_probe_broadband", "LaBraM probe / 0.1-75 Hz"),
        ("LaBraM_fullFT_broadband", "LaBraM full fine-tune / 0.1-75 Hz"),
    ):
        subjects = broad["results"][key]
        if len(subjects) != 10 or {r["subject"] for r in subjects} != expected:
            raise ValueError(f"{key}: missing or duplicate held-out subjects")
        if any(not 0 <= r["acc"] <= 1 for r in subjects):
            raise ValueError(f"{key}: accuracy outside [0, 1]")
        summary = {}
        for metric in ("acc", "kappa"):
            values = [r[metric] for r in subjects]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = pstdev(values)
        for metric, value in summary.items():
            if not math.isclose(value, broad["summary"][key][metric], abs_tol=1e-9):
                raise ValueError(f"{key}: stored {metric} does not match per-subject results")
        rows.append({"model": label, **summary})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true", help="Print machine-readable summaries")
    args = parser.parse_args()
    try:
        rows = load_results(args.data_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Cannot summarize saved results: {exc}\n")
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print("Split A: 10 held-out subjects. Saved results; no new training run.\n")
    for row in rows:
        print(f"{row['model']:<40} {100*row['acc_mean']:5.1f}% +/- {100*row['acc_std']:4.1f}%  kappa {row['kappa_mean']:+.3f}")
    print("\n+/- is population SD across subjects, not a confidence interval or variation across seeds.")
    print("Narrowband rows use the stored rounded summaries; broadband summaries are recomputed.")
    print("Broadband EEGNet is absent from split A; multi-split full-fine-tuning replication is incomplete.")


if __name__ == "__main__":
    main()
