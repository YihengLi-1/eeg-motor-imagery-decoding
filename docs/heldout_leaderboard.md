# Held-out leaderboard for EEG foundation models (reference implementation)

Updated 2026-10-31. Maintainer: Yiheng Li. Companion to the [pretraining-overlap table](pretraining_overlap_eeg_fm.md) and to the `datasets:` model-card metadata being added to `braindecode/*` checkpoints (braindecode PR #1493).

**What is different here.** Every number is computed by us, not self-reported, under one subject-independent protocol, and every cell carries a provenance badge saying whether that checkpoint's pretraining corpus contained the evaluation dataset. A score is only a cross-subject result when the badge is **[H]**.

Badges: **[H]** held out (dataset not in the checkpoint's stated pretraining corpus) · **[I]** in corpus · **[P]** same collection (a TUH sub-corpus while pretraining used TUEG) · **[R]** related release (e.g. SEED-V vs SEED) · **[S]** trained from scratch, no pretraining · **[?]** corpus not itemised by the authors.

## Motor imagery, binary, accuracy (mean ± SD over 4 subject-level splits)

| Checkpoint / training | PhysioNet MI (109 subj., left/right hand) | Cho2017 (52 subj., left/right hand) |
|---|---|---|
| EEGNet (from scratch) | 54.9 ± 1.0 [S] | 52.3 ± 0.6 [S] |
| TimeMIL (from scratch) | – | 59.2 ± 1.8 [S] |
| LaBraM-base, linear probe | 48.5 ± 2.3 [I] | 51.7 ± 0.5 [H] |
| LaBraM-base, full fine-tune | 79.1 ± 1.7 [I] | 65.8 ± 2.3 [H] |
| CBraMod, linear probe | 67.0 ± 3.3 [H] | 60.2 ± 1.7 [H] |
| CBraMod, full fine-tune | 82.3 ± 1.0 [H] | 71.4 ± 2.8 [H] |

**Protocol.** Test subjects are never seen in fine-tuning: 4 random subject-level splits (seeds 0–3); 0.1–75 Hz band, 4-s windows; Adam with early stopping on a validation set of held-out subjects; identical windows, splits and budget for every row. LaBraM and CBraMod start from the released braindecode weights; EEGNet and TimeMIL (InceptionTime backbone with time-aware MIL pooling, same splits and early stopping) are trained from scratch. Cho2017 amplitudes are rescaled by one global constant to PhysioNet's range for the pretrained models (EEGNet is scale-invariant). Code: `step3d_matrix.py`; raw results: `step3c_fullmatrix_results.json`, `heldout/results_cho2017_matrix.json`.

**How to read it.** LaBraM's PhysioNet number is **[I]**: PhysioNet (EEGMMIDB) is listed in LaBraM's pretraining corpus, so 79.1% is not a cross-subject result for that checkpoint. Its Cho2017 number is **[H]**. CBraMod pretrains on TUEG only, so both of its numbers are **[H]**; it drops by a similar amount between the two datasets (82.3 → 71.4) as LaBraM (79.1 → 65.8), which is why most of the gap is the dataset, not the overlap (see the paper draft for the direct test).

## Rows being added

- TimeMIL on PhysioNet MI
- BIOT (3 checkpoints)
- EEGPT
- CodeBrain
- BENDR
- REVE
- MIREPNet (PhysionetMI and Cho2017 are in its pretraining)

## Adding a checkpoint

Open an issue with the Hub repo id. We run it under the protocol above and take the provenance badge from the checkpoint's model card `datasets:` field (or from the audit table when the card has none). Authors who disagree with a badge: the audit table records page-cited quotes; send the correction and we update both.
