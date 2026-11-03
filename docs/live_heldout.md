# Live held-out bench for EEG foundation models (draft, 2026-10-31; season-1 results added 2026-11-01)

Companion to the [held-out leaderboard](heldout_leaderboard.md) and the [pretraining-overlap table](pretraining_overlap_eeg_fm.md). Maintainer: Yiheng Li.

## Why

Every public EEG dataset will eventually be inside somebody's pretraining corpus. The overlap table can tell you *which* checkpoint saw *which* dataset today, but it cannot manufacture an unseen test set. Time can. A dataset whose first public snapshot post-dates the upload of a checkpoint cannot have been in that checkpoint's pretraining data, whatever its model card says. So each quarter we take the EEG datasets that became public during that quarter, run every earlier public checkpoint on them under the one subject-independent protocol used throughout this repository, and publish the numbers with the dates that make them held-out by construction.

This is the LiveBench idea (Whiteet al., 2024, for language models) applied to EEG. Nobody runs it for neural data yet.

## The rule

What the time rule does and does not guarantee: it rules out the dataset having reached the pretraining corpus through its public release. It does not rule out private sharing before release, an earlier alias of the same recordings, or the Hub upload date being later than the real training cutoff (we use the earlier of Hub upload and paper weights release). It is a stronger, dated statement than "not listed in the corpus", not a proof.

A result on dataset *D* for checkpoint *M* is **[T] held out by time** when

> first public snapshot date of *D*  >  upload date of *M* on the Hub (or the release date of the weights the Hub copy was converted from, whichever is later).

Both dates are recorded in the results file. Datasets are also checked against the overlap table and against the EEGDash Hub mirror (which republishes OpenNeuro datasets for pretraining pipelines); a dataset that appears in the mirror before a checkpoint's upload is not **[T]** for that checkpoint.

## Protocol (unchanged)

0.1–75 Hz, resampled to 200 Hz, 5-s windows (1000 samples; for cued tasks −1 to +4 s around the cue, as on the leaderboard); 4 subject-level random splits (seeds 0–3; ~25 % of subjects test, ~15 % validation, rest train); Adam with early stopping on validation balanced accuracy; identical windows, splits and budget for every row. Models: EEGNet from scratch; LaBraM-base linear probe and full fine-tune; CBraMod linear probe and full fine-tune. Code: `heldout/live_heldout.py`.

## Checkpoint upload dates used for the time rule (braindecode Hub org, queried 2026-10-31)

| Checkpoint | first upload |
|---|---|
| braindecode/labram-pretrained | 2025-12-19 (weights: LaBraM, ICLR 2024) |
| braindecode/cbramod-pretrained | 2026-01-12 (weights: CBraMod, ICLR 2025) |
| braindecode/eegpt-pretrained | 2026-01-06 |
| braindecode/SignalJEPA* | 2025-10-28 / 2025-12-19 |
| braindecode/codebrain-pretrained | 2026-04-03 |
| braindecode/REVE | 2026-04-25 |
| braindecode/STEEGFormer-* | 2026-06-24 |
| braindecode/brant-pretrained | 2026-07-09 |
| braindecode/mvpformer-* | 2026-07-16 |
| braindecode/brainbert-pretrained / popt-pretrained | 2026-07-20 / 2026-07-24 |
| braindecode/SleepFM | 2026-07-30 |
| braindecode/ZUNA | 2026-08-01 (latest) |

EEGDash Hub mirror of OpenNeuro: last batch created 2026-04-20; none of the datasets below is in it.

## Season 2026-Q3/Q4 candidates (OpenNeuro EEG datasets first published 2026-07-01 … 2026-10-31)

| Dataset | first snapshot | subjects | task / label | size | files on S3 | status |
|---|---|---|---|---|---|---|
| ds008446 Random and Sequential Finger MI (UCL) | 2026-07-21 | 20 | 5-finger motor imagery, 62 ch g.tec 512 Hz | 5.8 GB | yes | **season 1 done, table below** |
| ds008711 RSVP sentences with violations (SDSU) | 2026-08-20 | 64 | violation vs control ERP; event codes undocumented, deaf/hearing groups | 6.7 GB | yes | downloading; labels to be confirmed with authors |
| ds008763 Resting EEG, autistic traits (Chongqing Normal) | 2026-09-01 | 80 | high vs low AQ-10 (subject-level) | 6.2 GB | yes | candidate |
| ds008851 Alcohol cue-reactivity (Neuphony cap) | 2026-09-21 | 60 | AUD vs control, emotion | 0.2 GB | yes | candidate, consumer cap |
| ds008858 MI mouse-cursor (Emotiv 14 ch) | 2026-09-23 | 18 | left/right MI vs rest | 0.15 GB | **403** (not exported yet) | retry |
| ds008859 MI mouse-cursor (OpenBCI 8 ch) | 2026-09-23 | 7 | 6-class MI | 0.02 GB | yes | too small |
| ds008875 Resting EEG, Parkinson's vs controls (Iowa) | 2026-09-26 | 252 | PD / control / DBS (subject-level) | 10 GB | **403** (not exported yet) | retry; strongest clinical set |
| ds008892 CuBiAAD auditory attention | 2026-09-30 | 30 EEG | attended side | 1.9 GB | **403** | retry |
| ds008730 Flavors of Reward Positivity | 2026-08-26 | 104 | win/loss feedback ERP (doors) | 47 GB | yes | candidate, large |
| ds008968 Spell-IT-Out covert/overt speech (Padova) | 2026-10-20 | 25 | syllable / word decoding | 53 GB | yes | candidate, large |
| ds008768 Resting EEG PD vs HC | 2026-09-01 | 312 | PD / HC | 64 GB | yes | **excluded**: authors ask not to use pending revalidation |

Datasets whose first snapshot is before 2026-08-01 are **[T]** for every checkpoint except ZUNA (and before 2026-07-30 also SleepFM); the per-checkpoint rule above handles this automatically.

## Season 1 plan

1. ds008446, 5-class finger MI, 20 subjects: EEGNet, LaBraM probe/FT, CBraMod probe/FT, 4 seeds. Report accuracy, balanced accuracy and kappa (chance 20 %).
2. One subject-level clinical set (ds008875 when exported, otherwise ds008763).
3. Publish the season table here and in the leaderboard, with both dates per cell. Ask the dataset authors to confirm our label derivation before publishing (same courtesy as the overlap table).

## Season 1 results: ds008446 (run 2026-11-01)

Data as used: 20 subjects in the release; **16 kept**. Runs whose channels were nearly all flagged by the amplitude/flatness check were skipped (sub-11, sub-13, sub-14, sub-15: 55–62 of 62 channels flagged in every run, so the subject has no usable run; three single runs of sub-07/08/09 also skipped). Raw EDF amplitudes are in the thousands of µV before filtering; after 8–30 Hz filtering the median channel SD is 18 µV. Interpolated channels per kept run: median 13, max 30 of 62. Labels: five fingers of the same hand (thumb, index, middle, ring, pinky), both the random and sequential tasks pooled; chance 20 %. Protocol as above. Both checkpoints were uploaded before the dataset's first snapshot (2026-07-21), so every pretrained row is **[T]**.

| model | accuracy % (4 seeds) | balanced acc. % | kappa | tag |
|---|---|---|---|---|
| EEGNet, from scratch | 28.6 ± 3.9 | 28.7 ± 3.9 | +0.108 ± 0.049 | [S] |
| LaBraM-base, linear probe | 20.9 ± 1.8 | 20.9 ± 1.8 | +0.012 ± 0.023 | [T] |
| LaBraM-base, full fine-tune | 23.9 ± 4.9 | 23.9 ± 4.9 | +0.049 ± 0.062 | [T] |
| CBraMod, linear probe | 21.4 ± 1.1 | 21.4 ± 1.1 | +0.018 ± 0.013 | [T] |
| CBraMod, full fine-tune | 31.8 ± 4.3 | 31.8 ± 4.3 | +0.148 ± 0.054 | [T] |

Per-seed accuracies: EEGNet 34.6 / 25.7 / 29.6 / 24.6; LaBraM probe 19.6 / 19.8 / 24.0 / 20.3; LaBraM FT 31.5 / 25.2 / 19.2 / 19.9; CBraMod probe 20.7 / 22.6 / 22.3 / 20.1; CBraMod FT 38.3 / 28.8 / 33.0 / 27.2. Raw results: `heldout/results_live_ds008446.json`.

Reading: five-finger imagery across subjects is barely above chance for every model; both linear probes are at chance, so neither frozen representation carries finger information that transfers across subjects. Fine-tuning CBraMod gives the best number and is the only row clearly above the from-scratch EEGNet, by about 3 points with overlapping seed spread. Fine-tuned LaBraM is below EEGNet, as on PhysioNet and Cho2017. All EEGNet rows here were produced after the MPS grouped-convolution fix (`docs/mps_groupconv_bug.md`).

## Known limits

- New datasets are small and heterogeneous (consumer caps, odd montages). A season reports what exists; it does not pick datasets for flattering numbers.
- Compute is one laptop (M3 Pro). We run five model configurations, not every Hub checkpoint; authors may request a run through the issue template.
- Time guarantees the dataset is unseen; it does not guarantee the *subjects* are new (a group may republish earlier participants). We record this when the dataset says so.
