# Held-out leaderboard for EEG foundation models (reference implementation)

Updated 2026-10-31. Maintainer: Yiheng Li. Companion to the [pretraining-overlap table](pretraining_overlap_eeg_fm.md) and to the `datasets:` model-card metadata being added to `braindecode/*` checkpoints (braindecode PR #1493).

**What is different here.** Every number is computed by us, not self-reported, under one subject-independent protocol, and every cell carries a provenance badge saying whether that checkpoint's pretraining corpus contained the evaluation dataset. Fine-tuning is subject-independent for every row; the badge records a second, separate dimension: whether the dataset appears in the checkpoint's stated pretraining corpus. **[H]** means it does not appear in the disclosed corpus, not that the test subjects are independently verified unseen.

Badges: **[H]** held out (dataset not in the checkpoint's stated pretraining corpus) · **[I]** in corpus · **[P]** same collection (a TUH sub-corpus while pretraining used TUEG) · **[R]** related release (e.g. SEED-V vs SEED) · **[S]** trained from scratch, no pretraining · **[?]** corpus not itemised by the authors.

## Motor imagery, binary, accuracy (mean ± SD over 4 subject-level splits)

| Checkpoint / training | PhysioNet MI (109 subj., left/right hand) | Cho2017 (52 subj., left/right hand) |
|---|---|---|
| EEGNet (from scratch) | 54.9 ± 1.0 [S] | 52.3 ± 0.6 [S] |
| TimeMIL (from scratch) | – | 59.2 ± 1.6 [S] |
| Riemannian tangent space + LR (OAS covariance; classical, no deep net) | – | 57.1 ± 2.5 [S] (0.1–75 Hz); 55.4 ± 2.3 [S] (8–30 Hz) |
| LaBraM-base, linear probe | 48.5 ± 2.3 [I] | 51.7 ± 0.5 [H] |
| LaBraM-base, full fine-tune | 79.1 ± 1.7 [I] | 65.8 ± 2.3 [H] |
| CBraMod, linear probe | 67.0 ± 3.3 [H] | 60.2 ± 1.7 [H] |
| CBraMod, full fine-tune | 82.3 ± 1.0 [H] | 71.4 ± 2.8 [H] |

**Protocol.** Test subjects are never seen in fine-tuning: 4 random subject-level splits (seeds 0–3); 0.1–75 Hz band, 5-s windows (−1 to 4 s around the cue, 1000 samples at 200 Hz); Adam with early stopping on a validation set of held-out subjects; identical windows, splits and budget for every row. LaBraM and CBraMod start from the released braindecode weights; EEGNet and TimeMIL (InceptionTime backbone with time-aware MIL pooling, same splits and early stopping) are trained from scratch. The Riemannian row is the standard classical pipeline (shrinkage covariance per window, tangent-space mapping with the reference point fitted on training subjects only, standardisation, logistic regression; C chosen on the validation subjects), added 2026-10-31 as the strong non-deep baseline so that the foundation-model gain is not measured only against EEGNet. Code: `heldout/riemann_cho2017.py`; raw results: `heldout/results_cho2017_riemann.json`. Cho2017 amplitudes are rescaled by one global constant to PhysioNet's range for the pretrained models (EEGNet is scale-invariant). Code: `step3d_matrix.py`; raw results: `step3c_fullmatrix_results.json`, `heldout/results_cho2017_matrix.json`.

**How to read it.** LaBraM's PhysioNet number is **[I]**: PhysioNet (EEGMMIDB) is listed in LaBraM's pretraining corpus, so 79.1% is not a cross-subject result for that checkpoint. Its Cho2017 number is **[H]**. CBraMod pretrains on TUEG only, so both of its numbers are **[H]**; it drops by a similar amount between the two datasets (82.3 → 71.4) as LaBraM (79.1 → 65.8). That is consistent with the dataset being harder rather than with an overlap effect, but two different models can react differently to a dataset change, so this comparison is a hint, not a bound; the direct tests are the peek intervention in the paper draft and the random-initialisation rows below.

## Rows being added

- LaBraM architecture from random initialisation and CBraMod from random initialisation, same full-training recipe as the fine-tuned rows (running 2026-10-31): the direct control for what the pretrained weights contribute, which neither EEGNet nor TimeMIL can give because they change the architecture.
- Season 1 of the [time-split live bench](live_heldout.md) (ds008446, first published 2026-07-21, after the LaBraM and CBraMod Hub uploads used here; the time rule is applied per checkpoint).

- TimeMIL on PhysioNet MI
- BIOT (3 checkpoints)
- EEGPT
- CodeBrain
- BENDR
- REVE
- MIREPNet (PhysionetMI and Cho2017 are in its pretraining)

## Adding a checkpoint

Open an issue with the Hub repo id. We run it under the protocol above and take the provenance badge from the checkpoint's model card `datasets:` field (or from the audit table when the card has none). Authors who disagree with a badge: the audit table records page-cited quotes; send the correction and we update both.
