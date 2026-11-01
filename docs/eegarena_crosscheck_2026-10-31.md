# Cross-check: EEG-Arena's pretraining-overlap exclusions vs. this registry (2026-10-31)

EEG-Arena (Chen, Peng, Qin et al., arXiv 2609.32743; eeg.fm) excludes a model–task cell from its rankings when "a model's pretraining data directly overlap with the corresponding downstream dataset" (red "E" badge). Its public data file `benchmarks-data/overall-performance.json` (version 1, release 2026-09-15; source CSV `figure2a_overall_performance_matrix_direct_overlap_cells.csv`) marks 228 of 5,720 cells. The paper and site do not publish the per-model pretraining lists behind the badges. We compared every badge for the models that both EEG-Arena and this registry cover.

## Result in one line

For the ten shared models, every EEG-Arena "E" badge agrees with a dataset named in the model's own pretraining list as recorded here, and we found no badge that our evidence contradicts. The difference is one of policy: EEG-Arena marks **direct** overlap (the evaluation dataset itself is named in the pretraining list); this registry additionally records **same-collection** overlap (the evaluation dataset is a sub-corpus of a named pretraining corpus, e.g. TUEV and TUSL inside TUEG) together with each paper's exclusion statement or control run.

## Direct overlap (EEG-Arena "E"): agreement

| EEG-Arena model | datasets marked E | registry evidence (page-cited in `pretraining_overlap_eeg_fm.md`) |
|---|---|---|
| LaBraM | BCIC-IV-1, Kaggle-INRIA, PhysioNet-MI, SEED, TUSL | all five named in LaBraM's pretraining list (Table of corpora: BCI Competition IV-1, Inria BCI Challenge, EEGMMIDB, SEED series, TUSL) |
| EEGPT (Wang et al.) | HGD, PhysioNet-MI, SEED | named (PhysioMI, HGD, SEED) |
| REVE | BCIC-IV-2B, HGD, OpenBMI, Stroke-MI | named in Appendix B (BNCI2014004, Schirrmeister2017, Lee2019MI, Liu2024) |
| NeuroRVQ | BCIC-IV-1, Kaggle-INRIA, PhysioNet-MI | named in Appendix E, Table 9 (BCI Competition IV-1, Inria BCI Challenge, Physionet MI) |
| ST-EEGFormer | BCIC-IV-2A, BCIC-IV-2B, HGD, Large-MI | batch-2 audit of the braindecode checkpoints: BNCI2014_001 and other MOABB sets in the in-house pretraining set |

Not checked (model not yet in this registry): HEAR (9 badges), mdJPT (FACED, SEED), LEAD, Uni-DMFM, SleepFM, SleepGPT, TFM, EEGMamba.

## Same-collection overlap: cells EEG-Arena ranks that this registry marks PARTIAL

TUEV and TUSL are sub-corpora of TUEG. Models pretrained on TUEG (or on other TUEG sub-corpora) are evaluated on both in EEG-Arena without a badge. What each paper says:

| model | TUEG in pretraining | paper's statement about TUEV / TUSL | registry label |
|---|---|---|---|
| CBraMod | whole TUEG | re-pretrained without TUEV as a control: BAcc 0.6659 vs 0.6671 (App. E.7) | PARTIAL, control available |
| CodeBrain | whole TUEG | re-pretrained removing TUEV and TUAB samples: BAcc 0.6375 vs 0.6428 (p. 30–31) | PARTIAL, control available |
| LUNA | TUEG + Siena | TUAB/TUAR/TUSL subjects and recordings "strictly excluded" (p. 6) | PARTIAL, exclusion statement |
| LaBraM | TUAR, TUEP, TUSZ, TUSL (not whole TUEG) | "recordings of TUAB and TUEV are disjoint from recordings of pre-training datasets" (Sec. 3.5) | PARTIAL (TUEV), IN (TUSL, which EEG-Arena does badge) |
| CSBrain | whole TUEG | no exclusion statement or control found | PARTIAL, no statement |
| Neuro-GPT | TUH corpus (~20,000 recordings) | paper does not evaluate on TUEV/TUSL; no statement | PARTIAL, no statement |
| BENDR | TUEG v1.1/1.2 | paper does not evaluate on TUEV/TUSL; no statement | PARTIAL, no statement |
| REVE | whole TUEG | "those used in downstream tasks" removed at recording level (Sec. 3.1.1); TUEV is a REVE downstream set, TUSL is not mentioned | PARTIAL |

Whether these cells should be ranked is a policy choice for the benchmark; the registry's role is to make the evidence behind the choice visible per model.

## One inconsistency between two public sources (not an EEG-Arena issue)

NeuralBench's REVE page (facebookresearch/neuroai, `docs/neuralbench/models/reve.rst`) lists "PhysionetMI" among REVE's pretraining datasets that overlap NeuralBench tasks. REVE's paper (arXiv 2510.21585, Appendix B) names 25 of its 27 MOABB datasets, none of them PhysionetMI, uses PhysioNet-MI as a downstream task, and states that recordings used in downstream tasks were removed from pretraining. EEG-Arena does not badge REVE on PhysioNet-MI, consistent with the paper. One of the two Meta/REVE sources needs a correction; we will ask the REVE authors.

## What this registry could add for EEG-Arena

1. A cited, versioned source for the existing badges (JSON, one record per model × dataset, with page quotes and the author-confirmation status), so the badge logic is reproducible by others.
2. A second, clearly separated annotation for same-collection cells, carrying each paper's exclusion statement or control, which the benchmark can rank or not as it chooses.
3. Rows for the EEG-Arena models not yet in the registry, added by the same procedure (paper quotes, then author confirmation).

Sample records in `registry_sample_eegarena.json`.
