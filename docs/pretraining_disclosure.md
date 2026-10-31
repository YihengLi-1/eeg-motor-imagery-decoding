# Pretraining disclosure for biosignal foundation-model checkpoints

Version 0.1, 2026-11-01. Maintained with the pretraining-overlap audit. Comments: open an issue.

A checkpoint's downstream result is only a cross-subject result if the evaluation data was not in the checkpoint's pretraining data. That can be checked only if the pretraining data is declared in a form a machine can compare against evaluation datasets. This page says what to declare and where.

## 1. Where

In the checkpoint's Hugging Face model card. Two YAML fields in the front matter, and one prose section.

```yaml
datasets:          # what the weights were pretrained on (and, for fine-tuned weights, fine-tuned on)
  - EEGDash/ds002718
  - PhysionetMI
  - TUEG
base_model: braindecode/labram-pretrained   # only for weights derived from another checkpoint
```

`datasets:` is read by `huggingface_hub.ModelCard.load(repo_id).data.datasets`, indexed by the Hub as `dataset:<id>` tags, and (from braindecode 1.9) exposed as `model.card_data.datasets` after `from_pretrained`.

## 2. Naming

One identifier per dataset, chosen in this order:

1. The Hub dataset id when the dataset is hosted on the Hub (OpenNeuro sets mirrored by EEGDash: `EEGDash/ds002718`; others as published).
2. Otherwise the MOABB or braindecode dataset class name: `PhysionetMI`, `BNCI2014_001`, `Cho2017`, `Lee2019_MI`, `Schirrmeister2017`, `Stieger2021`.
3. Otherwise the short name the community already uses: `TUEG`, `TUAB`, `TUEV`, `SEED`, `SEED-IV`, `SHHS`, `Siena`, `HBN-EEG`, `BrainTreebank`.

Do not invent new names when one of the above exists. Do not put a collection and its sub-corpora both in the list unless both were used as separate sources; `TUEG` implies TUAB, TUEV, TUSL, TUAR, TUSZ and TUEP.

## 3. What goes in `datasets:` and what does not

In: every dataset whose recordings entered pretraining, public or gated, named as above.

Not in the field, but stated in the prose section:

- **Private or self-collected data**: give hours, subjects and the collecting institution. Example: "plus 342 h of self-collected EEG from more than 140 subjects (ESI NeuroScan, 62 channels), not public."
- **Un-itemised aggregates**: if the paper says "56 OpenNeuro datasets" or "208 datasets" without listing them, say so and list what is known. Readers must be able to tell "not listed" from "not used".
- **Exclusions**: if downstream subjects, sessions or recordings were removed from pretraining, say which and how (subject-level or recording-level). Example: "every TUH sub-corpus used downstream was filtered out of TUEG pretraining at the recording level."
- **Splits**: for datasets where only part was used (first 40 of 54 subjects; 19 of 26 sessions), give the split.
- **Fine-tuned checkpoints**: name the fine-tuning data and labels, and point `base_model:` at the parent.

## 4. Prose section template

```
## Pretraining data

Sources (hours, subjects): TUEG (27,062 h, 14,987 subj.); Siena (30 h, 14 subj.) ...
Private or un-itemised: ...
Excluded from pretraining: ...
Splits: ...
Source of this list: <paper section/page or author statement, date>
```

## 5. How the declaration is used

- Evaluation: `set(model.card_data.datasets) & {evaluation dataset ids}` gives the in-corpus set; a TUH sub-corpus against `TUEG` is same-collection.
- Registry: the audit table records the same information with page-cited quotes and an author-verified flag; cards and table are kept consistent, and a correction in one is propagated to the other.
- Competitions: a submission's method description can carry the same two fields, which makes "declare every external corpus" checkable against the evaluation sets.

## 6. Verification

A row or card is marked **author-verified** when an author of the model confirms it in writing (email or issue). Verified rows may carry the badge:

[![pretraining provenance: checked](https://img.shields.io/badge/pretraining%20provenance-checked-2ea44f)](https://github.com/YihengLi-1/eeg-motor-imagery-decoding/blob/main/docs/pretraining_overlap_eeg_fm.md)
