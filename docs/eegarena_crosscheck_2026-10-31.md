# Cross-check: EEG-Arena's pretraining-overlap badges vs. this registry (2026-10-31)

EEG-Arena (Chen, Peng, Qin et al., arXiv 2609.32743; eeg.fm) excludes a model–task cell from its rankings when "a model's pretraining data directly overlap with the corresponding downstream dataset" (red "E" badge). Its public data file `https://eeg.fm/benchmarks-data/overall-performance.json` (version 1, releaseDate 2026-09-15, source CSV `figure2a_overall_performance_matrix_direct_overlap_cells.csv`, fetched 2026-10-31) marks 228 of 5,720 cells. In the materials we checked (the paper, the benchmark pages and this file) we did not find a per-cell source for the badges, so we compared them with the evidence in this registry.

## Scope

- Models in both EEG-Arena and this registry (the intersection): BENDR, BIOT, LaBraM, EEGPT (Wang et al.), CBraMod, LUNA, REVE, CSBrain, NeuroGPT, CodeBrain. These account for 1,200 evaluated cells and 55 of the 228 badge cells (LaBraM 17, EEGPT 14, REVE 24; the other seven carry no badge).
- Checked in addition from the per-checkpoint audit of the braindecode Hub (not in the main registry table): NeuroRVQ and ST-EEGFormer.
- Not checked (not yet in the registry): HEAR, mdJPT, LEAD, Uni-DMFM, SleepFM, SleepGPT, TFM, EEGMamba.
- What a check means here: for every badge in scope, is the evaluation dataset named in the model's own pretraining description as recorded in this registry (page-cited quotes)? We did not independently recompute EEG-Arena's totals and we cannot verify cells without a badge beyond what the papers state.

## Badges in scope: all agree with a named pretraining dataset

| model | datasets badged by EEG-Arena | registry evidence (quotes and pages in `pretraining_overlap_eeg_fm.md`) |
|---|---|---|
| LaBraM | BCIC-IV-1, Kaggle-INRIA, PhysioNet-MI, SEED, TUSL | the 16-corpus pretraining list names BCI Competition IV-1, Inria BCI Challenge, EEGMMIDB, the SEED series and TUSL |
| EEGPT (Wang et al.) | HGD, PhysioNet-MI, SEED | pretraining list names PhysioMI, HGD, SEED |
| REVE | BCIC-IV-2B, HGD, OpenBMI, Stroke-MI | Appendix B names BNCI2014004, Schirrmeister2017, Lee2019MI, Liu2024 |
| NeuroRVQ (supplementary) | BCIC-IV-1, Kaggle-INRIA, PhysioNet-MI | arXiv 2510.13068 v4, Appendix E, Table 9: BCI Competition IV-1, Inria BCI Challenge, Physionet MI |
| ST-EEGFormer (supplementary) | BCIC-IV-2A, BCIC-IV-2B, HGD, Large-MI | braindecode checkpoints `STEEGFormer-small/base/large/largeV2`: in-house pretraining set that includes BNCI2014_001 and other MOABB motor-imagery sets (see batch-2 notes in PR #1493) |

We found no badge in scope that the registry's evidence contradicts.

## What the registry records in addition: same-collection relation and exposure evidence

TUEV and TUSL are sub-corpora of the TUH EEG Corpus (TUEG). Sharing a parent collection does not by itself mean the evaluation recordings were in pretraining; what matters is the version and filtering each paper reports. For the TUEG-family models that EEG-Arena evaluates on TUEV and TUSL, the registry holds:

| model | TUEG use as reported | what the paper says about TUEV / TUSL | registry label |
|---|---|---|---|
| CBraMod | TUEG (69,652 recordings, 14,987 subjects, 27,062 h after filtering) | control: re-pretrained without TUEV, BAcc 0.6659 vs 0.6671 (App. E.7) | PARTIAL, control available |
| CodeBrain | TUEG (1,109,545 samples, about 9,246 h) | control: re-pretrained without TUEV and TUAB samples, BAcc 0.6375 vs 0.6428 (pp. 30–31) | PARTIAL, control available |
| LUNA | TUEG (14,987 subjects, 21,787 h) + Siena | TUAB, TUAR, TUSL subjects and recordings "strictly excluded" (p. 6); author-confirmed 2026-10-24 | PARTIAL with exclusion (no badge is consistent with the statement) |
| LaBraM | TUAR, TUEP, TUSZ, TUSL (sub-corpora, not the whole TUEG) | "recordings of TUAB and TUEV are disjoint from recordings of pre-training datasets" (Sec. 3.5) | TUEV PARTIAL with statement; TUSL IN (and badged by EEG-Arena) |
| CSBrain | TUEG (1,109,545 segments, over 9,000 h) | no exclusion statement or control found | PARTIAL, no statement |
| Neuro-GPT | TUH corpus (about 20,000 recordings) | TUEV/TUSL not evaluated in the paper; no statement | PARTIAL, no statement |
| BENDR | TUEG v1.1 and v1.2 | TUEV/TUSL not evaluated in the paper; no statement | PARTIAL, no statement |
| REVE | TUH (14,987 subjects, 26,847 h) | recordings "used in downstream tasks" removed (Sec. 3.1.1); TUEV is a REVE downstream set, TUSL is not mentioned | PARTIAL with statement |

Whether a benchmark ranks these cells is its own decision. The registry's contribution is to keep the relation (same collection) separate from the exposure evidence (control run, exclusion statement, or nothing) so that the decision can be made per model and revisited when authors add information.

## Two public sources that need reconciliation (not an EEG-Arena issue)

NeuralBench's REVE page (`facebookresearch/neuroai`, `docs/neuralbench/models/reve.rst`, "Pretraining data overlap") lists PhysionetMI among REVE's pretraining datasets overlapping the `motor_imagery` task. REVE's paper (arXiv 2510.21585 v1, Appendix B) names 25 of its 27 MOABB pretraining datasets, none of them PhysionetMI, uses PhysioNet-MI as a downstream task, and states (Sec. 3.1.1) that recordings used in downstream tasks were removed from pretraining. EEG-Arena's absence of a badge for REVE on PhysioNet-MI is compatible with the paper's statement. The two unnamed MOABB datasets, the checkpoint version NeuralBench evaluates, and the level of the exclusion (recording or subject) would settle it; we will ask the REVE authors and the NeuralBench maintainers separately.

## What this registry could offer a benchmark

1. A versioned, cited source for existing badges: one record per checkpoint × dataset with the paper location and quote, the exclusion level if any, and the scope and date of any author confirmation.
2. A separate annotation for same-collection cells carrying each paper's control or statement, to rank or exclude as the benchmark chooses.
3. Rows for models the registry does not yet cover, added by the same procedure.

Sample records: `registry_sample_eegarena.json`.
