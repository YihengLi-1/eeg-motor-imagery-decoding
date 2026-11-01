# Live held-out bench for EEG foundation models (draft, 2026-10-31)

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
| ds008446 Random and Sequential Finger MI (UCL) | 2026-07-21 | 20 | 5-finger motor imagery, 62 ch g.tec 512 Hz | 5.8 GB | yes | **season 1, running** |
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

## Known limits

- New datasets are small and heterogeneous (consumer caps, odd montages). A season reports what exists; it does not pick datasets for flattering numbers.
- Compute is one laptop (M3 Pro). We run five model configurations, not every Hub checkpoint; authors may request a run through the issue template.
- Time guarantees the dataset is unseen; it does not guarantee the *subjects* are new (a group may republish earlier participants). We record this when the dataset says so.
