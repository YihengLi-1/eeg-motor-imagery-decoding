# EEG foundation models: pretraining corpus vs. downstream benchmark overlap

Compiled 2026-10-24 by Yiheng Li from the primary PDFs (arXiv / proceedings). Machine-readable twin: [`pretraining_overlap_eeg_fm.json`](pretraining_overlap_eeg_fm.json).

**How to read this.** This table records whether each reported downstream result uses a dataset that also appears, fully or as a sub-corpus, in the model's own stated pretraining data. An overlap label is not a claim of test-label leakage: pretraining is self-supervised, and several authors state that they removed downstream recordings or subjects from pretraining (marked `*`). The point is to let readers tell in-corpus results from held-out ones. Corrections are welcome via an issue or PR.

**Author verification.** Rows confirmed by the model authors are marked here and in the JSON. FEMBA, LUNA and LuMamba (ETH Zürich, Integrated Systems Laboratory) were confirmed on 2026-10-24; the authors add that every downstream TUH dataset used for fine-tuning was filtered out of their TUH pretraining data (e.g. LUNA fine-tuned on TUAB never saw TUAB in pretraining), consistent with the `*` marks below. CodeBrain and Uni-NTFM were confirmed by their corresponding author (Ziyu Jia) on 2026-10-26. Corrections from other authors are welcome via an issue or PR.

**Labels.** IN-CORPUS = same dataset listed in the model's own pretraining corpus. PARTIAL = downstream set is a subset or derivative of a pretraining corpus (TUAB, TUEV, TUSL, TUAR, TUSZ are all TUH/TUEG sub-corpora; LaBraM Appendix D and CBraMod Appendix E call TUEV, TUAB, TUAR, TUSL 'a subset of TUEG'). HELD-OUT = not in the pretraining list. UNCLEAR = list missing or identity ambiguous. In the tables: I, P, H, ? are those four; `*` marks 'authors claim they removed it from pretraining'; `r` marks a same-family sibling dataset that is in pretraining.

**Page convention.** `p.N` is the page index of the downloaded PDF. Quotes are copied from the PDF text with line breaks and hyphenation normalised (an automated check confirmed each quoted string is on the cited page, +-1).

## 1. Headline numbers

- Models audited: 20. Gram (ICASSP 2025) could not be audited (no open full text found).
- Reported (model, EEG dataset) results: 142. IN-CORPUS: 1. PARTIAL: 32. HELD-OUT: 92. UNCLEAR: 17.
- Of the 32 PARTIAL results: 4 have a re-pretrain-without-it control run (CBraMod, CodeBrain), 13 have only a textual exclusion claim (LaBraM, FEMBA, LUNA, REVE, LuMamba), 15 have neither (NeuroLM, CSBrain, EEGFormer, Uni-NTFM, EEG-DINO).
- Models with at least one IN/PARTIAL result: 12 of 20. Fully clean models: BENDR, BIOT, EEGPT (Wang et al.), Neuro-GPT, BrainOmni, LCM (Large Cognition Model).
- 15 of 20 audited models pretrain on TUEG or TUEG sub-corpora (BENDR, LaBraM, CBraMod, NeuroLM, FEMBA, LUNA, REVE, CSBrain, EEGFormer, Neuro-GPT, CodeBrain, Uni-NTFM, EEG-DINO, LuMamba, BrainWave (EEG + iEEG)). Any of them evaluated on TUAB/TUEV/TUSL/TUAR/TUSZ is PARTIAL by construction.

## 2. Summary table

| Model | Pretraining corpora (own list) | Downstream results, mark per dataset | I / P / H / ? |
|---|---|---|---|
| **BENDR** (Frontiers in Human Neuroscience 2021) | TUEG v1.1 and v1.2 (about 1.5 TB EDF before preprocessing) | MMI [H], BCIC [H], ERN [H], P300 [H], SSC [H] | 0 / 0 / 5 / 0 |
| **BIOT** (NeurIPS 2023) | SHHS (5,445 sleep recordings, C3-A2/C4-A1, used without labels); PREST (private hospital sleep-lab EEG, 6,478 recordings; institution not named); Cardiology (five public ECG sets, ECG not EEG) | CHB-MIT [H], IIIC Seizure [H], TUAB [H], TUEV [H] | 0 / 0 / 4 / 0 |
| **LaBraM** (ICLR 2024) | BCI Competition IV-1 (8.21 h); Emobrain (4.94 h); Grasp and Lift EEG Challenge (11.72 h); Inria BCI Challenge (29.98 h) [= Kaggle ERN data]; EEG Motor Movement/Imagery Dataset = PhysioNet EEGMMIDB (47.3 h); Raw EEG Data, Trujillo 2020 (34.35 h); Resting State EEG Data, Trujillo 2017 (3.04 h); SEED series: SEED, SEED-IV, SEED-... | TUAB [P*], TUEV [P*], SEED-V [Hr], MoBI gait regression [H] | 0 / 2 / 2 / 0 |
| **EEGPT (Wang et al.)** (NeurIPS 2024) | PhysioMI = EEGMMIDB (109 subj); HGD (14 subj); TSU SSVEP benchmark (35 subj); SEED (15 subj); M3CV (106 subj) | BCIC-2A [H], BCIC-2B [H], Sleep-EDFx [H], KaggleERN [H], PhysioP300 [H], TUAB [H], TUEV [H] | 0 / 0 / 7 / 0 |
| **CBraMod** (ICLR 2025) | TUEG only (69,652 recordings, 14,987 subjects, 27,062 h; after cleaning 1,109,545 samples, >9,000 h) | FACED [H], SEED-V [H], PhysioNet-MI [H], SHU-MI [H], ISRUC [H], CHB-MIT [H], BCIC2020-3 [H], Mumtaz2016 [H], SEED-VIG [H], MentalArithmetic [H], TUEV [P*], TUAB [P*], BCIC-IV-2a [H] | 0 / 2 / 11 / 0 |
| **NeuroLM** (ICLR 2025) | TUEG (about 24,000 h); SEED Series: SEED-IV (15), SEED-V (20), SEED-GER (8), SEED-FRA (8) (170.54 h); BCI Competition IV-1 (8.21 h); Emobrain (4.94 h); Grasp and Lift (11.72 h); Inria BCI (29.98 h); Motor Movement/Imagery = EEGMMIDB (47.3 h); Raw EEG Data, Trujillo 2020 (34.35 h); Resting State, Trujillo 2017 (3.04 h); Siena ... | TUAB [P], TUEV [P], SEED [?r], HMC [H], Workload [H], TUSL [P] | 0 / 3 / 2 / 1 |
| **FEMBA** (EMBC 2025) | TUEG with subjects present in TUSL, TUAR, or TUAB filtered out (about 21,000 h, >5,000 participants in abstract; Table I lists 14,987 subjects) | TUAB [P*], TUAR [P*], TUSL [P*] | 0 / 3 / 0 / 0 |
| **LUNA** (NeurIPS 2025) | TUEG (14,987 subjects, 21,787.32 h); Siena Scalp EEG (14 subjects, 141.0 h) | TUAB [P*], TUAR [P*], TUSL [P*], SEED-V [H] | 0 / 3 / 1 / 0 |
| **REVE** (arXiv 2510.21585) | 92 datasets, 24,274 subjects, 61,415 h: TUH (14,987 subj, 26,847 h; full TUEG), PhysioNet (Siena, ICARE), 27 MOABB sets (e.g. BNCI2014004 = BCI-IV-2b, Schirrmeister2017 = HGD, Cho2017, Lee2019, Weibo2014 ...), 56 OpenNeuro sets (accession numbers only), Others (NMT, HMS, SparrKULee, Inria Large, THINGS2, TDBRAIN) | PhysioNet-MI [H], BCIC-IV-2a [H], TUEV [P*], TUAB [P*], HMC [H], ISRUC [H], FACED [H], Mumtaz [H], Mental Arithmetic [H], BCI2020-IV-3 [H] | 0 / 2 / 8 / 0 |
| **CSBrain** (arXiv 2506.23075) | TUEG only (1,109,545 segments, over 9,000 h after the CBraMod pipeline) | BCIC-IV-2a [H], PhysioNet-MI [H], SHU-MI [H], FACED [H], SEED-V [H], CHB-MIT [H], Siena [H], ISRUC [H], HMC [H], BCIC2020-3 [H], SEED-VIG [H], MentalArithmetic [H], Mumtaz2016 [H], TUEV [P], TUAB [P], TUSL [P] | 0 / 3 / 13 / 0 |
| **EEGFormer** (arXiv 2401.10278) | TUH EEG Corpus, '1.7TB of unlabelled EEG data' | TUAB [P], TUAR [P], TUSL [P], TUSZ [P], Neonate [H] | 0 / 4 / 1 / 0 |
| **Neuro-GPT** (arXiv 2311.03764) | TUH EEG corpus (about 20,000 recordings processed) | BCI Competition IV 2a [H] | 0 / 0 / 1 / 0 |
| **CodeBrain** (ICLR 2026) | TUH EEG Corpus (TUEG), 1,109,545 samples (about 9,246 h) | FACED [H], SEED-V [H], ISRUC S1 [H], ISRUC S3 [H], BCIC 2020-T3 [H], Mental Arithmetic [H], CHB-MIT [H], SHU-MI [H], TUEV [P*], TUAB [P*] | 0 / 2 / 8 / 0 |
| **BrainOmni** (arXiv 2505.18185) | 22 open EEG/MEG datasets, 1,997 h EEG + 656 h MEG: MEG-MASC, MEG-Narrative, OMEGA, CC700, Go-Nogo, MusicEEG, HFO, AversiveMEG, SRM, MIND, RestCog, SMN4Lang, HBN EO/EC, THINGS-MEG, ASWR-MEG, ImageLine, Features-EEG, PEARL-Neuro, NeuroMorph, Kymata-SOTO, HBN-EEG, Awakening. No TUH. | AD65 [H], MDD [H], PD31 [H], TUAB [H], TUEV [H], FACED [H], WBCIC_SHU [H], PhysioNet-MI [H], SomatoMotor [H] | 0 / 0 / 9 / 0 |
| **Uni-NTFM** (ICLR 2026) | Emobrain (4.94 h); REEG-BACA (121.6 h, 608 subj); SEED series: SEED-IV, SEED-V, SEED-GER, SEED-FRA (170.54 h, 51 subj); CAUEEG (306 h, 1388 subj); TUEG (27,100 h, 14,987 subj); Raw EEG Data, Trujillo 2020 (34.35 h); BCI Competition IV-1 (8.21 h); Resting State EEG, Trujillo 2017 (3.04 h); Siena Scalp EEG (30.47 h) | TUAB [P], TUEV [P], SEED [Ir], TDBrain [H], ADFTD [H], BCIC-IV-2a [H], Workload [H], HMC [H], TUSL [P] | 1 / 3 / 5 / 0 |
| **EEG-DINO** (MICCAI 2025) | TUEG (1,109,545 samples, over 9000 h; CBraMod pipeline) | TUEV [P], TUAB [P], SEED-V [H] | 0 / 2 / 1 / 0 |
| **LuMamba** (arXiv 2603.19100) | TUEG (about 21,600 h, >14,000 patients); the five downstream datasets 'strictly excluded' from pretraining | TUAB [P*], TUSL [P*], TUAR [P*], TDBrain [H], APAVA [H] | 0 / 3 / 2 / 0 |
| **LCM (Large Cognition Model)** (arXiv 2502.17464) | PhysioMI = EEGMMIDB; TSU (SSVEP); SEED | BCIC-2A [H], BCIC-2B [H] | 0 / 0 / 2 / 0 |
| **BrainWave (EEG + iEEG)** (arXiv 2402.10251) | EEG: CAP, HMC, Siena, SRM, TUEG, Schizophrenia-81, Sleep-EDF, Stroke-50, PD-31, IowaDataset, UNMDataset, AD-184, plus a private EEG corpus (35,675.5 h, 15,906 subjects); iEEG: CCEP and a private corpus (5,231 h, 91 subjects) | AD-65 [H], CHB-MIT [H], Absence-16 [?], Clonic-6 [?], Atonic-5 [?], Mayo-Clinic [H], FNUSA [H], DRE-Clinical [?], MDD-64 [H], Depression-122 [H], Schizophrenia-28 [H], ADHD-Adult [H], ADHD-Child [H], SD-71 [H] | 0 / 0 / 10 / 4 |
| **EEGPT/BrainGPT (Yue et al.)** (arXiv 2410.19779 v2) | Not enumerated: '37.5M pre-training samples' from 'diverse EEG datasets' | DEAP [?], FACED [?], SEED-IV [?], SEED-V [?], MIBCI [?], BCI Competition IV-1 [?], EEGMat [?], STEW [?], EDF [?], HMC [?], IMG [?], SPE [?] | 0 / 0 / 0 / 12 |

## 3. Per-model notes

### BENDR

Source: arXiv 2101.12037 | Frontiers in Human Neuroscience 2021 (arXiv preprint) | pretraining size: not stated in hours

**Pretraining corpus (as listed by the authors):**
- TUEG v1.1 and v1.2 (about 1.5 TB EDF before preprocessing)

**Quotes:**
- p.5: "The closest publicly accessible dataset, to our current knowledge, was the Temple University Hospital EEG Corpus (TUEG) [46]."
- p.5: "We focused specifically on versions 1.1 and 1.2 of this dataset which amounted to approximately 1.5 TB of European-data-format (EDF) EEG recordings before preprocessing."
- p.6 (downstream): "Table 1: Summary of downstream dataset battery and number of cross-validation folds used."

| Downstream dataset | Label | Comment |
|---|---|---|
| MMI (PhysioNet EEG Motor Movement/Imagery, 105 subjects) | HELD-OUT |  |
| BCIC (BCI Competition IV, 9 subjects, 4-class MI = 2a) | HELD-OUT |  |
| ERN (Inria BCI Challenge, 26 subjects) | HELD-OUT |  |
| P300 (Citi/Poli/Cinel Donchin speller, PhysioNet) | HELD-OUT |  |
| SSC (Sleep-EDFx, 83 subjects, Fpz-Cz/Pz-Oz) | HELD-OUT |  |

**Notes.** Pretraining is TUEG only. None of the five downstream sets is a TUH derivative, so all are HELD-OUT. (Reverse direction: the ERN set is the Inria BCI Challenge data that LaBraM and NeuroLM later put into their own pretraining corpora.)

**Headline results.** No in-corpus headline results.

### BIOT

Source: arXiv 2305.10351 | NeurIPS 2023 | pretraining size: about 5M SHHS samples (30 s) + 5M PREST samples (10 s)

**Pretraining corpus (as listed by the authors):**
- SHHS (5,445 sleep recordings, C3-A2/C4-A1, used without labels)
- PREST (private hospital sleep-lab EEG, 6,478 recordings; institution not named)
- Cardiology (five public ECG sets, ECG not EEG)

**Quotes:**
- p.5: "In the experiments, we pre-train an unsupervised encoder using 5 million resting EEG samples (16 channels, 10s, 200Hz) and 5 million sleep EEG samples (2 channels, 30s, 125Hz)"
- p.6: "The first three datasets are used entirely for unsupervised pre-training."
- p.13: "PREST is a private dataset recorded in hospital sleep lab, primarily for seizure and abnormal EEG detection purpose (such as spikes)."
- p.6 (downstream): "The next four datasets are used for supervised learning"
- p.9 (exclusion / leakage): "We obtain a Pre-trained (6 EEG datasets) model by loading the Pre-trained (PREST+SHHS) model and further train it on the training sets of CHB-MIT, IIIC Seizure, TUAB, and TUEV."

| Downstream dataset | Label | Comment |
|---|---|---|
| CHB-MIT (patients 1-19 train, 20-21 val, 22-23 test, then flipped) | HELD-OUT |  |
| IIIC Seizure (Ge et al. 2021; Jing et al. 2023) | HELD-OUT |  |
| TUAB (official split) | HELD-OUT | Not in the unsupervised pretraining corpora. See supervised variant below. |
| TUEV (official split) | HELD-OUT | Same as TUAB. |
| PTB-XL (ECG) | HELD-OUT [non-EEG] |  |
| HAR (wearable sensors) | HELD-OUT [non-EEG] |  |

**Notes.** Unsupervised BIOT variants (PREST, PREST+SHHS) are clean on every downstream set. The 'Pre-trained BIOT (6 EEG datasets)' variant (Sec 3.6, Tables 2/4/5) is further trained on the labeled TRAIN splits of CHB-MIT, IIIC, TUAB and TUEV before fine-tuning, so for that variant all four are IN-CORPUS at the train-split level (test subjects stay disjoint by the official TUAB/TUEV split and by patient split for the others). Sec 3.5 also does supervised pretraining on CHB-MIT/IIIC/TUAB then tests on TUEV; the paper itself notes 'TUAB and TUEV are both recorded from Temple University and share some common information'. PREST is private; the paper does not say which hospital, so overlap with the downstream sets cannot be checked.

**Headline results.** Unsupervised-pretrained headline numbers are clean. The 6-EEG-dataset variant uses same-dataset train splits.

### LaBraM

Source: arXiv 2405.18765 | ICLR 2024 | pretraining size: 2534.78 h total

**Pretraining corpus (as listed by the authors):**
- BCI Competition IV-1 (8.21 h)
- Emobrain (4.94 h)
- Grasp and Lift EEG Challenge (11.72 h)
- Inria BCI Challenge (29.98 h) [= Kaggle ERN data]
- EEG Motor Movement/Imagery Dataset = PhysioNet EEGMMIDB (47.3 h)
- Raw EEG Data, Trujillo 2020 (34.35 h)
- Resting State EEG Data, Trujillo 2017 (3.04 h)
- SEED series: SEED, SEED-IV, SEED-GER, SEED-FRA (166.75 h)
- Siena Scalp EEG Database (30.47 h)
- SPIS Resting State (0.83 h)
- Target Versus Non-Target / Brain Invaders (16 h)
- TUAR (92.22 h, subset of TUEG)
- TUEP (591.22 h, subset of TUEG)
- TUSZ (1138.53 h)
- TUSL (20.59 h, subset of TUEG)
- Self-collected EEG (342.23 h, >140 subjects, 62 ch, 1000 Hz, ESI NeuroScan; not itemised)

**Quotes:**
- p.17: "Training datasets (for both vector-quantized neural spectrum prediction training and LaBraM pre-training, the total time is 2534.78 hours):"
- p.18: "TUAR (Buckwalter et al., 2021): This subset of TUEG contains annotations of 5 different artifacts"
- p.18: "TUSL (von Weltin et al., 2017): This is another subset of TUEG that contains annotations of slowing events"
- p.6: "TUEV (event type classification) (Obeid & Picone, 2016): This corpus is a subset of TUEG"
- p.6 (downstream): "We systematically evaluate our LaBraM on the following downstream datasets:"
- p.19 (downstream): "We conduct two additional BCI tasks on the following datasets:"
- p.7 (exclusion / leakage): "Note that the four downstream datasets are excluded from the pre-training datasets."
- p.9 (exclusion / leakage): "It is noted that the recordings of TUAB and TUEV are disjoint from recordings of pre-training datasets."

| Downstream dataset | Label | Comment |
|---|---|---|
| TUAB | PARTIAL | Same TUH corpus family (TUAR/TUEP/TUSL/TUSZ, all TUEG derivatives, are in pretraining). Authors say TUAB recordings are disjoint from pretraining recordings and show an ablation (Fig. 4). Exclusion claim: Sec 3.2: 'the four downstream datasets are excluded from the pre-training datasets'; Sec 3.5: 'the recordings of TUAB and TUEV are disjoint from recordings of pre-training datasets'; Fig. 4 compares with/without adding TUAB/TUEV to pretraining (no significant change). |
| TUEV | PARTIAL | Same as TUAB. Exclusion claim: Same statements as TUAB. |
| SEED-V (Appendix F) | HELD-OUT (related sibling in corpus) | SEED-V itself is not listed, but SEED, SEED-IV, SEED-GER, SEED-FRA and 342 h of un-itemised self-collected 62-ch/1000 Hz NeuroScan EEG are. The paper cannot rule out overlap with SEED-V subjects. |
| MoBI gait regression (Appendix F) | HELD-OUT |  |

**Notes.** Pretraining list is Appendix D (about 20 datasets). TUAB and TUEV are the two main-text tasks (Tables 1-2). The excluded-claim is recording-level; TUH does not guarantee patient-level disjointness across sub-corpora, and the paper does not state patient-level checks. LaBraM's own pretraining includes EEGMMIDB, Siena, SEED, Inria BCI Challenge and BCI-IV-1, which later papers (BENDR-protocol: MMI/ERN; CBraMod/CSBrain/REVE: PhysioNet-MI) use as downstream benchmarks.

**Headline results.** Tables 1-2 (TUAB, TUEV) are PARTIAL: same-institution TUH sub-corpora (TUAR, TUEP, TUSL, TUSZ) are in the pretraining set; authors claim recording-level disjointness.

### EEGPT (Wang et al.)

Source: NeurIPS proceedings PDF (no arXiv version found) | NeurIPS 2024 | pretraining size: not stated in hours

**Pretraining corpus (as listed by the authors):**
- PhysioMI = EEGMMIDB (109 subj)
- HGD (14 subj)
- TSU SSVEP benchmark (35 subj)
- SEED (15 subj)
- M3CV (106 subj)

**Quotes:**
- p.6: "Table 1: Datasets for pretraining and downstream tasks"
- p.2: "EEGPT, with over 10 million parameters, is pretrained on a mixed multi-task EEG dataset, including data from PhysioMI [23], HGD [24], and M3CV [25]."
- p.17: "LaBraM employs a larger pretraining dataset than ours, which also contains the TUEG dataset that is similar to the TUAB and TUEV distributions"
- p.6 (downstream): "It contains MI datasets BCIC-2A [39], BCIC-2B [40], sleep stage detection dataset Sleep-EDFx [11] , error related negativity (ERN) dataset KaggleERN [41] and event-related potentials dataset PhysioP300 [23]."

| Downstream dataset | Label | Comment |
|---|---|---|
| BCIC-2A | HELD-OUT |  |
| BCIC-2B | HELD-OUT |  |
| Sleep-EDFx | HELD-OUT |  |
| KaggleERN | HELD-OUT |  |
| PhysioP300 (PhysioNet erpbci) | HELD-OUT | Different PhysioNet dataset from PhysioMI despite the shared reference number [23] in the paper. |
| TUAB | HELD-OUT |  |
| TUEV | HELD-OUT |  |

**Notes.** Table 1 (p6) lists five pretraining sets; the intro (p2) names only PhysioMI, HGD, M3CV (internal inconsistency; Table 1 and Appendix C.1 list all five). No TUH data in pretraining, so TUAB/TUEV are HELD-OUT here, and the appendix says LaBraM benefits from TUEG. All downstream sets are clean.

**Headline results.** All headline results are HELD-OUT.

### CBraMod

Source: arXiv 2412.07236 | ICLR 2025 | pretraining size: >9000 h after cleaning

**Pretraining corpus (as listed by the authors):**
- TUEG only (69,652 recordings, 14,987 subjects, 27,062 h; after cleaning 1,109,545 samples, >9,000 h)

**Quotes:**
- p.6: "CBraMod is pre-trained on a very large public dataset, Temple University Hospital EEG corpus (TUEG) (Obeid & Picone, 2016)."
- p.3: "The majority of downstream datasets are sourced from institutions different from the pretraining dataset."
- p.7: "Table 1: Overview of downstream BCI tasks and datasets."
- p.7 (downstream): "Table 1: Overview of downstream BCI tasks and datasets."
- p.26 (exclusion / leakage): "Given that the TUEV dataset is a subset of the pretraining dataset TUEG, we introduced an additional experimental setup, CBraMod (excluding TUEV), to eliminate potential effects of data leakage."
- p.27 (exclusion / leakage): "Considering that the TUAB dataset is a subset of the pretraining dataset TUEG, we implemented an additional experimental setup, CBraMod (excluding TUAB), to mitigate potential data leakage effects."

| Downstream dataset | Label | Comment |
|---|---|---|
| FACED | HELD-OUT |  |
| SEED-V | HELD-OUT |  |
| PhysioNet-MI | HELD-OUT |  |
| SHU-MI | HELD-OUT |  |
| ISRUC | HELD-OUT |  |
| CHB-MIT | HELD-OUT |  |
| BCIC2020-3 | HELD-OUT |  |
| Mumtaz2016 | HELD-OUT |  |
| SEED-VIG | HELD-OUT |  |
| MentalArithmetic | HELD-OUT |  |
| TUEV | PARTIAL | Control run: re-pretrained without TUEV. BAcc 0.6659 vs 0.6671 (full), kappa 0.6744 vs 0.6772, wF1 0.8331 vs 0.8342. Exclusion claim: App. E.7 (p26): 'Given that the TUEV dataset is a subset of the pretraining dataset TUEG, we introduced an additional experimental setup, CBraMod (excluding TUEV), to eliminate potential effects of data leakage.' |
| TUAB | PARTIAL | Control run: re-pretrained without TUAB. BAcc 0.8249 vs 0.8289, AUC-PR 0.9221 vs 0.9258, AUROC 0.9156 vs 0.9227. Exclusion claim: App. E.8 (p27): 'Considering that the TUAB dataset is a subset of the pretraining dataset TUEG, we implemented an additional experimental setup, CBraMod (excluding TUAB), to mitigate potential data leakage effects.' |
| BCIC-IV-2a (App. E.9, Tables 15 and 23) | HELD-OUT |  |

**Notes.** Only TUEV and TUAB are TUH-derived. The control is a clean ablation (dataset removed and model re-pretrained) but it is reported only in the appendix tables (13, 14); the main-text Table 1/summary numbers use the full-TUEG model. Differences are small (<=0.004 BAcc, 0.007 AUROC).

**Headline results.** 2 of 13 reported datasets (TUEV, TUAB) are PARTIAL; both have an exclusion control with small deltas.

### NeuroLM

Source: arXiv 2409.00101 | ICLR 2025 | pretraining size: about 25,000 h

**Pretraining corpus (as listed by the authors):**
- TUEG (about 24,000 h)
- SEED Series: SEED-IV (15), SEED-V (20), SEED-GER (8), SEED-FRA (8) (170.54 h)
- BCI Competition IV-1 (8.21 h)
- Emobrain (4.94 h)
- Grasp and Lift (11.72 h)
- Inria BCI (29.98 h)
- Motor Movement/Imagery = EEGMMIDB (47.3 h)
- Raw EEG Data, Trujillo 2020 (34.35 h)
- Resting State, Trujillo 2017 (3.04 h)
- Siena Scalp EEG (30.47 h)
- SPIS Resting State (0.83 h)
- Target vs Non-Target (16 h)
- Self-collected EEG (342.23 h)

**Quotes:**
- p.15: "The total time after data cleaning is close to 25,000 hours."
- p.16: "Table 6: Information of datasets used for pre-training."
- p.7: "About 25,000 hours of EEG data from multiple public EEG datasets are collected after cleaning and filtering, which are listed in Appendix C."
- p.6 (downstream): "We consider six different EEG datasets with highly varied data sizes to comprehensively evaluate NeuroLM"

| Downstream dataset | Label | Comment |
|---|---|---|
| TUAB | PARTIAL | Entire TUEG (about 24,000 h) is in pretraining; no exclusion or control reported. |
| TUEV | PARTIAL | Same as TUAB. |
| SEED (3-class, 15 subjects) | UNCLEAR (related sibling in corpus) | Pretraining table lists 'SEED Series' = SEED-IV, SEED-V, SEED-GER, SEED-FRA (51 subjects, 170.54 h). The original SEED (Zheng & Lu 2015) is the downstream set but is not itemised. LaBraM's list (166.75 h) names SEED explicitly. Cannot tell from the paper whether original SEED is inside. |
| HMC (sleep, 151 subjects) | HELD-OUT |  |
| Workload (Zyma 2019, 36 subjects) | HELD-OUT |  |
| TUSL | PARTIAL | Entire TUEG in pretraining; no exclusion or control reported. |

**Notes.** Searched the full text for exclude / leak / overlap / disjoint: no statement that TUAB, TUEV or TUSL recordings were removed from the TUEG pretraining set. NeuroLM also multi-task instruction-tunes on the downstream training splits (labels), which is supervised exposure rather than unlabeled test-subject exposure.

**Headline results.** 3 of 6 headline tasks (TUAB, TUEV, TUSL) are PARTIAL with no control; SEED is UNCLEAR.

### FEMBA

Source: arXiv 2502.06438 | EMBC 2025 (arXiv 2502.06438) | pretraining size: about 21,000 h

**Pretraining corpus (as listed by the authors):**
- TUEG with subjects present in TUSL, TUAR, or TUAB filtered out (about 21,000 h, >5,000 participants in abstract; Table I lists 14,987 subjects)

**Quotes:**
- p.4: "We pre-train FEMBA on the TUEG dataset, as detailed in Section II-A. To prevent data leakage between pre-training and downstream tasks, we use a version of TUEG where subjects present in TUSL, TUAR, or TUAB have been filtered out."
- p.3: "TABLE I: Summary of Datasets Used"
- p.4 (downstream): "We assess FEMBA on three downstream tasks using the datasets described in Section II-B."
- p.4 (exclusion / leakage): "To prevent data leakage between pre-training and downstream tasks, we use a version of TUEG where subjects present in TUSL, TUAR, or TUAB have been filtered out."

| Downstream dataset | Label | Comment |
|---|---|---|
| TUAB | PARTIAL | Subject-level filter applied to the pretraining set (same corpus, same hospital). Exclusion claim: p4: subjects present in TUSL, TUAR or TUAB filtered out of TUEG pretraining. |
| TUAR | PARTIAL | Same filter. Exclusion claim: Same. |
| TUSL | PARTIAL | Same filter. Exclusion claim: Same. |

**Notes.** All three downstream sets are TUH sub-corpora, so every reported result is same-source. The paper states a subject-level filter, which is the strongest exclusion claim among TUEG-pretrained models; TUEV is not evaluated. Abstract says '21,000 hours of unlabeled EEG from 5,000 subjects' while Table I lists 14,987 TUEG subjects.

**Headline results.** All 3 reported datasets are TUH derivatives (PARTIAL), with a subject-level exclusion claim.

### LUNA

Source: arXiv 2510.22257 | NeurIPS 2025 | pretraining size: over 21,900 h

**Pretraining corpus (as listed by the authors):**
- TUEG (14,987 subjects, 21,787.32 h)
- Siena Scalp EEG (14 subjects, 141.0 h)

**Quotes:**
- p.6: "We pre-train LUNA on a combined corpus of Temple University Hospital EEG Corpus (TUEG) [9] and the Siena Scalp EEG Database [10]"
- p.6: "All subjects and recordings from the downstream evaluation datasets (TUAB, TUAR, TUSL, SEED-V) were strictly excluded from this pre-training set to ensure fair evaluation of generalization."
- p.23: "Table 11: Summary of Datasets Used."
- p.6 (downstream): "Downstream evaluations cover four diverse benchmarks"
- p.6 (exclusion / leakage): "All subjects and recordings from the downstream evaluation datasets (TUAB, TUAR, TUSL, SEED-V) were strictly excluded from this pre-training set to ensure fair evaluation of generalization."

| Downstream dataset | Label | Comment |
|---|---|---|
| TUAB | PARTIAL | Subjects and recordings excluded from pretraining by the authors' statement. Exclusion claim: p6: 'strictly excluded'. |
| TUAR | PARTIAL | Same. Fine-tuning uses an 80/10/10 sample-level random split (not subject-disjoint) as in EEGFormer. Exclusion claim: p6. |
| TUSL | PARTIAL | Same. Sample-level random split. Exclusion claim: p6. |
| SEED-V | HELD-OUT | Authors call it an 'unseen 62-channel topology'. |

**Notes.** Exclusion is stated at subject-and-recording level. Separate issue: TUAR and TUSL fine-tuning splits are sample-level random, so train/test overlap in subjects exists inside the downstream task itself (authors acknowledge this on p6).

**Headline results.** 3 of 4 reported datasets are TUH derivatives (PARTIAL) with a strong exclusion claim; SEED-V is HELD-OUT.

### REVE

Source: arXiv 2510.21585 | arXiv 2510.21585 (preprint) | pretraining size: 61,415 h

**Pretraining corpus (as listed by the authors):**
- 92 datasets, 24,274 subjects, 61,415 h: TUH (14,987 subj, 26,847 h; full TUEG), PhysioNet (Siena, ICARE), 27 MOABB sets (e.g. BNCI2014004 = BCI-IV-2b, Schirrmeister2017 = HGD, Cho2017, Lee2019, Weibo2014 ...), 56 OpenNeuro sets (accession numbers only), Others (NMT, HMS, SparrKULee, Inria Large, THINGS2, TDBRAIN)

**Quotes:**
- p.5: "It comprises 19 TB of raw data, spanning 24,274 subjects, 150,833 unique sessions, and 61,415 hours of recordings drawn from 92 different sources, including OpenNeuro (Markiewicz et al., 2021), MOABB (Aristimunha et al., 2023), and TUH (Obeid and Picone, 2016)."
- p.6: "We only removed recordings shorter than 10 seconds, and those used in downstream tasks."
- p.21: "Table 7: Detailed overview of the pretraining datasets."
- p.7: "Table 1: Overview of downstream tasks and datasets."
- p.7 (downstream): "across the following datasets: PhysioNet-MI (Goldberger et al., 2000), BCIC-IV-2a (Tangermann et al., 2012), TUEV (Obeid and Picone, 2016), TUAB (Obeid and Picone, 2016), HMC"
- p.6 (exclusion / leakage): "We only removed recordings shorter than 10 seconds, and those used in downstream tasks."

| Downstream dataset | Label | Comment |
|---|---|---|
| PhysioNet-MI | HELD-OUT | Not in the itemised MOABB/PhysioNet lists. Note: the few-shot 'XFT' variant (Table 22) fine-tunes on labeled MI sets that include (Goldberger et al., 2000). |
| BCIC-IV-2a | HELD-OUT | BNCI2014001 is not in the MOABB list; BNCI2014004 (=2b) is. |
| TUEV | PARTIAL | Full TUH corpus in pretraining; authors remove 'those used in downstream tasks' (recording level, wording ambiguous). Exclusion claim: p6: 'We only removed recordings shorter than 10 seconds, and those used in downstream tasks.' |
| TUAB | PARTIAL | Same. Exclusion claim: p6, same sentence. |
| HMC | HELD-OUT |  |
| ISRUC | HELD-OUT |  |
| FACED | HELD-OUT |  |
| Mumtaz | HELD-OUT |  |
| Mental Arithmetic (MAT) | HELD-OUT |  |
| BCI2020-IV-3 | HELD-OUT |  |

**Notes.** Removal of 'recordings used in downstream tasks' is not defined at subject level, and no TUEV/TUAB control run is reported. 56 OpenNeuro datasets are listed by accession number only, so name-level overlap with the downstream sets was checked only against datasets with names (none match).

**Headline results.** 2 of 10 downstream tasks (TUEV, TUAB) are PARTIAL; recording-level removal claimed, no control.

### CSBrain

Source: arXiv 2506.23075 | arXiv 2506.23075 (preprint) | pretraining size: >9000 h

**Pretraining corpus (as listed by the authors):**
- TUEG only (1,109,545 segments, over 9,000 h after the CBraMod pipeline)

**Quotes:**
- p.6: "We pre-train CSBrain on the Temple University Hospital EEG corpus (TUEG) dataset, which has been shown effective for foundation model studies [40, 57]."
- p.7: "After cleaning, a total of 1,109,545 EEG segments (over 9,000 hours) are used for pre-training."
- p.7: "Table 1: Overview of Downstream BCI datasets and tasks used for evaluation."
- p.7 (downstream): "Table 1: Overview of Downstream BCI datasets and tasks used for evaluation."

| Downstream dataset | Label | Comment |
|---|---|---|
| BCIC-IV-2a | HELD-OUT |  |
| PhysioNet-MI | HELD-OUT |  |
| SHU-MI | HELD-OUT |  |
| FACED | HELD-OUT |  |
| SEED-V | HELD-OUT |  |
| CHB-MIT | HELD-OUT |  |
| Siena | HELD-OUT |  |
| ISRUC | HELD-OUT |  |
| HMC | HELD-OUT |  |
| BCIC2020-3 | HELD-OUT |  |
| SEED-VIG | HELD-OUT |  |
| MentalArithmetic | HELD-OUT |  |
| Mumtaz2016 | HELD-OUT |  |
| TUEV | PARTIAL | Whole TUEG in pretraining. No exclusion or control found (searched leak/exclude/overlap/disjoint). |
| TUAB | PARTIAL | Same. |
| TUSL | PARTIAL | Same. |

**Notes.** Pipeline copied from CBraMod but without CBraMod's exclusion controls. 3 of 16 datasets are TUH derivatives and are all in the main Table 2.

**Headline results.** 3 of 16 datasets (TUEV, TUAB, TUSL) are PARTIAL with no control.

### EEGFormer

Source: arXiv 2401.10278 | arXiv 2401.10278 (preprint/workshop) | pretraining size: not stated in hours (1.7 TB)

**Pretraining corpus (as listed by the authors):**
- TUH EEG Corpus, '1.7TB of unlabelled EEG data'

**Quotes:**
- p.3: "We pretrain our model on the Temple University EEG Corpus (TUH Corpus)"
- p.1: "evaluating its performance on four downstream corpora sourced from the TUH corpus."
- p.3: "Notably, the Neonate dataset is not a subset of the TUH dataset."
- p.3 (downstream): "We evaluate our model on five downstream datasets."

| Downstream dataset | Label | Comment |
|---|---|---|
| TUAB | PARTIAL | 'sourced from the TUH corpus'; no exclusion stated. |
| TUAR | PARTIAL | Same. |
| TUSL | PARTIAL | Same. |
| TUSZ | PARTIAL | Same. |
| Neonate (Helsinki neonatal seizures, Stevenson 2019) | HELD-OUT | Authors state it is not a TUH subset; used as the transfer test. |

**Notes.** The paper frames the TUH-derived tasks as in-distribution and only the Neonate set as transfer. No leakage discussion.

**Headline results.** 4 of 5 reported datasets are TUH derivatives (PARTIAL); only Neonate is HELD-OUT.

### Neuro-GPT

Source: arXiv 2311.03764 | arXiv 2311.03764 (ISBI 2024 short paper) | pretraining size: not stated

**Pretraining corpus (as listed by the authors):**
- TUH EEG corpus (about 20,000 recordings processed)

**Quotes:**
- p.3: "Pre-training Dataset: The large-scale public dataset, Temple University Hospital (TUH) EEG corpus, is used as the pre-training dataset."
- p.3 (downstream): "Downstream Dataset: We define the downstream task as motor imagery classification, using the BCI Competition IV Dataset 2a"

| Downstream dataset | Label | Comment |
|---|---|---|
| BCI Competition IV 2a (9 subjects) | HELD-OUT |  |

**Notes.** Single downstream set, clean.

**Headline results.** Clean.

### CodeBrain

Source: arXiv 2506.09110 | ICLR 2026 (arXiv 2506.09110) | pretraining size: about 9,246 h

**Pretraining corpus (as listed by the authors):**
- TUH EEG Corpus (TUEG), 1,109,545 samples (about 9,246 h)

**Quotes:**
- p.7: "We pretrain CodeBrain on the TUH EEG Corpus (Obeid & Picone, 2016), the largest publicly available EEG dataset to date."
- p.30: "We note that both TUAB and TUEV originate from the TUH EEG corpus (Obeid & Picone, 2016), which overlaps with our pretraining source (TUEG)."
- p.7 (downstream): "Table 1: Summary of downstream tasks and associated EEG datasets."
- p.31 (exclusion / leakage): "we also report the results of experiments conducted by removing all TUEV and TUAB samples from the TUEG dataset."

| Downstream dataset | Label | Comment |
|---|---|---|
| FACED | HELD-OUT |  |
| SEED-V | HELD-OUT |  |
| ISRUC S1 | HELD-OUT |  |
| ISRUC S3 | HELD-OUT |  |
| BCIC 2020-T3 | HELD-OUT |  |
| Mental Arithmetic | HELD-OUT |  |
| CHB-MIT | HELD-OUT |  |
| SHU-MI | HELD-OUT |  |
| TUEV | PARTIAL | Control run removing all TUEV and TUAB samples from TUEG: BAcc 0.6375 vs 0.6428, kappa 0.6838 vs 0.6912. Pretraining stopped at epoch 10 'to avoid overfitting to this distribution'. Exclusion claim: p30-31: 'we also report the results of experiments conducted by removing all TUEV and TUAB samples from the TUEG dataset.' |
| TUAB | PARTIAL | Same control: BAcc 0.8288 vs 0.8294, AUROC 0.9012 vs 0.9030. Exclusion claim: Same. |

**Notes.** TUEV, TUAB, SHU-MI and ISRUC S1 results are in Appendix I (Tables 5-8), not the main comparison table.

**Headline results.** 2 of 10 (TUEV, TUAB) PARTIAL; both have an exclusion control; they sit in the appendix.

### BrainOmni

Source: arXiv 2505.18185 | arXiv 2505.18185 (preprint) | pretraining size: 1,997 h EEG + 656 h MEG

**Pretraining corpus (as listed by the authors):**
- 22 open EEG/MEG datasets, 1,997 h EEG + 656 h MEG: MEG-MASC, MEG-Narrative, OMEGA, CC700, Go-Nogo, MusicEEG, HFO, AversiveMEG, SRM, MIND, RestCog, SMN4Lang, HBN EO/EC, THINGS-MEG, ASWR-MEG, ImageLine, Features-EEG, PEARL-Neuro, NeuroMorph, Kymata-SOTO, HBN-EEG, Awakening. No TUH.

**Quotes:**
- p.5: "1997 hours of EEG data and 656 hours of MEG data collected from open-source datasets are used for pretraining (see Appendix H for details)"
- p.6: "To prevent information leakage, all datasets used for downstream testing were excluded from the pretraining process."
- p.6 (downstream): "Table 1: Overview of the downstream datasets."
- p.6 (exclusion / leakage): "To prevent information leakage, all datasets used for downstream testing were excluded from the pretraining process."

| Downstream dataset | Label | Comment |
|---|---|---|
| AD65 (Miltiadous, OpenNeuro ds004504) | HELD-OUT |  |
| MDD (Mumtaz 2016) | HELD-OUT |  |
| PD31 | HELD-OUT |  |
| TUAB | HELD-OUT |  |
| TUEV | HELD-OUT |  |
| FACED | HELD-OUT |  |
| WBCIC_SHU | HELD-OUT |  |
| PhysioNet-MI | HELD-OUT |  |
| SomatoMotor (EEG+MEG) | HELD-OUT |  |
| ASD74 (MEG) | HELD-OUT [non-EEG] |  |
| MEG-MMI (MEG) | HELD-OUT [non-EEG] |  |

**Notes.** Cleanest design among large models: no TUH in pretraining. Two extra sets (PerceiveImagine EEG, Gloups-MEG) are used only for zero-shot tokenizer generalisation (unseen devices).

**Headline results.** All headline results are HELD-OUT.

### Uni-NTFM

Source: arXiv 2509.24222 | ICLR 2026 (arXiv 2509.24222) | pretraining size: about 28,000 h

**Pretraining corpus (as listed by the authors):**
- Emobrain (4.94 h)
- REEG-BACA (121.6 h, 608 subj)
- SEED series: SEED-IV, SEED-V, SEED-GER, SEED-FRA (170.54 h, 51 subj)
- CAUEEG (306 h, 1388 subj)
- TUEG (27,100 h, 14,987 subj)
- Raw EEG Data, Trujillo 2020 (34.35 h)
- BCI Competition IV-1 (8.21 h)
- Resting State EEG, Trujillo 2017 (3.04 h)
- Siena Scalp EEG (30.47 h)

**Quotes:**
- p.7: "We aggregated a large-scale pre-training corpus by amalgamating nine distinct, publicly available EEG datasets"
- p.21: "Table 6: Information of datasets used for pre-training."
- p.22: "Table 7: Information of datasets used for downstream evaluation."
- p.7 (downstream): "we collected nine public EEG datasets across various paradigms and tasks"

| Downstream dataset | Label | Comment |
|---|---|---|
| TUAB | PARTIAL | TUEG (27,100 h) in pretraining; no exclusion or control found. |
| TUEV | PARTIAL | Same. |
| SEED | IN-CORPUS (related sibling in corpus) | Downstream is named 'SEED (Zheng & Lu, 2015)' but Table 7 gives 16 subjects and 5-class labels, which matches the open-source SEED-V, and the pretraining 'SEED-series' explicitly includes SEED-V (20 subjects) and SEED-IV. If the downstream set were the original 3-class SEED the label would be PARTIAL (family). Inference from Table 7 specs, not an explicit statement. |
| TDBrain | HELD-OUT |  |
| ADFTD | HELD-OUT |  |
| BCIC-IV-2a | HELD-OUT |  |
| Workload (Zyma 2019) | HELD-OUT |  |
| HMC | HELD-OUT |  |
| TUSL | PARTIAL | TUEG in pretraining; no exclusion found. |

**Notes.** Searched the full text for exclude / leak / disjoint / overlap: nothing about removing downstream data from TUEG. Pretraining list matches NeuroLM/LaBraM apart from CAUEEG and REEG-BACA.

**Headline results.** 4 of 9 downstream tasks (TUAB, TUEV, TUSL, SEED) overlap the pretraining corpus, none with a control.

### EEG-DINO

Source: MICCAI proceedings paper PDF | MICCAI 2025 (no arXiv version found) | pretraining size: >9000 h

**Pretraining corpus (as listed by the authors):**
- TUEG (1,109,545 samples, over 9000 h; CBraMod pipeline)

**Quotes:**
- p.5: "Pre-training Dataset We perform pre-training on the Temple University Hospital EEG (TUEG) dataset [11], one of the largest publicly available EEG datasets."
- p.5: "TUEV [11] is a curated subset of the TUEG database annotated for six typical events."
- p.5: "TUAB [11] represents a curated selection from the TUEG database"
- p.5 (downstream): "SEED-V [10] is a multimodal emotion recognition dataset featuring five emotions."

| Downstream dataset | Label | Comment |
|---|---|---|
| TUEV | PARTIAL | Authors call it a 'curated subset of the TUEG database'; no exclusion or control. |
| TUAB | PARTIAL | Same. |
| SEED-V (16 subjects) | HELD-OUT |  |

**Notes.** Authors state outright that the TUH downstream sets are subsets of the pretraining corpus and report no control.

**Headline results.** 2 of 3 datasets PARTIAL, no control.

### LuMamba

Source: arXiv 2603.19100 | arXiv 2603.19100 (preprint) | pretraining size: about 21,600 h

**Pretraining corpus (as listed by the authors):**
- TUEG (about 21,600 h, >14,000 patients); the five downstream datasets 'strictly excluded' from pretraining

**Quotes:**
- p.2: "We pretrain on the Temple University Hospital EEG (TUEG) corpus, comprising approximately 21,600 hours of recordings from more than 14,000 patients"
- p.3: "All five datasets are strictly excluded from the pretraining stage."
- p.3 (downstream): "LuMamba is evaluated on five EEG datasets: three TUH benchmarks (TUAB, TUSL, TUAR) and two datasets with unseen channel montages (TDBrain and APAVA)."
- p.3 (exclusion / leakage): "All five datasets are strictly excluded from the pretraining stage."

| Downstream dataset | Label | Comment |
|---|---|---|
| TUAB | PARTIAL | Excluded per authors. Exclusion claim: p3: 'strictly excluded from the pretraining stage'. |
| TUSL | PARTIAL | Same. 80/10/10 split. Exclusion claim: p3. |
| TUAR | PARTIAL | Same. 80/10/10 split. Exclusion claim: p3. |
| TDBrain (PD) | HELD-OUT |  |
| APAVA (AD) | HELD-OUT |  |

**Notes.** Same group and protocol as LUNA/FEMBA. Note that REVE (not LuMamba) has TDBRAIN in its pretraining corpus.

**Headline results.** 3 of 5 PARTIAL with an exclusion claim.

### LCM (Large Cognition Model)

Source: arXiv 2502.17464 | arXiv 2502.17464 (preprint) | pretraining size: not stated

**Pretraining corpus (as listed by the authors):**
- PhysioMI = EEGMMIDB
- TSU (SSVEP)
- SEED

**Quotes:**
- p.4: "Table 1: Datasets for pretraining and downstream tasks"
- p.4 (downstream): "This includes MI datasets BCIC-2A [14] and BCIC-2B [13]."

| Downstream dataset | Label | Comment |
|---|---|---|
| BCIC-2A | HELD-OUT |  |
| BCIC-2B | HELD-OUT |  |

**Notes.** Closely follows EEGPT's data table. Clean.

**Headline results.** Clean.

### BrainWave (EEG + iEEG)

Source: arXiv 2402.10251 | arXiv 2402.10251 (preprint) | pretraining size: 40,907 h total

**Pretraining corpus (as listed by the authors):**
- EEG: CAP, HMC, Siena, SRM, TUEG, Schizophrenia-81, Sleep-EDF, Stroke-50, PD-31, IowaDataset, UNMDataset, AD-184, plus a private EEG corpus (35,675.5 h, 15,906 subjects)
- iEEG: CCEP and a private corpus (5,231 h, 91 subjects)

**Quotes:**
- p.13: "The EEG recordings consisted of CAP [56], HMC [57], Siena [58], SRM [59], TUEG [60], Schizophrenia-81, Sleep-EDF [61], Stroke-50 [62], PD-31 [63], IowaDataset, UNMDataset, AD-184 [64], and a private EEG corpus"
- p.14 (downstream): "The evaluation benchmark comprised 15 distinct datasets, which contain 7 different healthcare scenarios."

| Downstream dataset | Label | Comment |
|---|---|---|
| AD-65 (Miltiadous, OpenNeuro ds004504) | HELD-OUT | Pretraining has AD-184, a different cohort. |
| CHB-MIT | HELD-OUT |  |
| Absence-16 (private) | UNCLEAR | Private; pretraining also includes unnamed private EEG/iEEG corpora. |
| Clonic-6 (private) | UNCLEAR | Same. |
| Atonic-5 (private) | UNCLEAR | Same. |
| Mayo-Clinic (iEEG) | HELD-OUT |  |
| FNUSA (iEEG) | HELD-OUT |  |
| DRE-Clinical (private iEEG) | UNCLEAR | Private; pretraining includes a private iEEG corpus. |
| MDD-64 | HELD-OUT |  |
| Depression-122 | HELD-OUT |  |
| Schizophrenia-28 | HELD-OUT | Pretraining has Schizophrenia-81, a different cohort. |
| ADHD-Adult | HELD-OUT |  |
| ADHD-Child | HELD-OUT |  |
| SD-71 | HELD-OUT |  |
| Apnea-ECG (ECG) | HELD-OUT [non-EEG] |  |

**Notes.** Clinical EEG/iEEG model (included because widely compared against LaBraM). No exclusion statement found. Private downstream sets cannot be checked against private pretraining corpora. Its pretraining includes HMC, Sleep-EDF and Siena, i.e. sets that other models evaluate on.

**Headline results.** No public downstream set is in-corpus; 4 private sets are UNCLEAR.

### EEGPT/BrainGPT (Yue et al.)

Source: arXiv 2410.19779 | arXiv 2410.19779 v2 (IEEE journal submission) | pretraining size: about 1B tokens

**Pretraining corpus (as listed by the authors):**
- Not enumerated: '37.5M pre-training samples' from 'diverse EEG datasets'

**Quotes:**
- p.2: "Based on it, we collect a total of 37.5M pre-training samples with approximately 1B tokens."
- p.5: "We evaluate our BrainGPT using 12 datasets across 5 distinct tasks, as detailed in Table I."
- p.5 (downstream): "For Emotion Recognition (ER), we utilize DEAP [29], FACED [30], SEED-IV [31], and SEED-V [32]."

| Downstream dataset | Label | Comment |
|---|---|---|
| DEAP | UNCLEAR | Pretraining datasets are not listed in the paper text, so overlap cannot be assessed. |
| FACED | UNCLEAR | Pretraining datasets are not listed in the paper text, so overlap cannot be assessed. |
| SEED-IV | UNCLEAR | Pretraining datasets are not listed in the paper text, so overlap cannot be assessed. |
| SEED-V | UNCLEAR | Pretraining datasets are not listed in the paper text, so overlap cannot be assessed. |
| MIBCI | UNCLEAR | Pretraining datasets are not listed in the paper text, so overlap cannot be assessed. |
| BCI Competition IV-1 | UNCLEAR | Pretraining datasets are not listed in the paper text, so overlap cannot be assessed. |
| EEGMat | UNCLEAR | Pretraining datasets are not listed in the paper text, so overlap cannot be assessed. |
| STEW | UNCLEAR | Pretraining datasets are not listed in the paper text, so overlap cannot be assessed. |
| EDF (Sleep-EDF) | UNCLEAR | Pretraining datasets are not listed in the paper text, so overlap cannot be assessed. |
| HMC | UNCLEAR | Pretraining datasets are not listed in the paper text, so overlap cannot be assessed. |
| IMG | UNCLEAR | Pretraining datasets are not listed in the paper text, so overlap cannot be assessed. |
| SPE | UNCLEAR | Pretraining datasets are not listed in the paper text, so overlap cannot be assessed. |

**Notes.** Different model from the NeurIPS 2024 EEGPT (Wang et al.). The v2 text gives only the sample count for pretraining. A full list may exist in a supplement or repo, which was not audited.

**Headline results.** Cannot be assessed (pretraining list missing).

## 4. Dataset-centric view (which downstream sets sit inside which models' pretraining)

| Downstream dataset | Evaluated by (own-model label) | Inside pretraining of (any audited model) |
|---|---|---|
| TUAB | BIOT [H], LaBraM [P], EEGPT (Wang et al.) [H], CBraMod [P], NeuroLM [P], FEMBA [P], LUNA [P], REVE [P], CSBrain [P], EEGFormer [P], CodeBrain [P], BrainOmni [H], Uni-NTFM [P], EEG-DINO [P], LuMamba [P] | BENDR (PARTIAL), LaBraM (PARTIAL), CBraMod (PARTIAL), NeuroLM (PARTIAL), FEMBA (PARTIAL), LUNA (PARTIAL), REVE (PARTIAL), CSBrain (PARTIAL), EEGFormer (PARTIAL), Neuro-GPT (PARTIAL), CodeBrain (PARTIAL), Uni-NTFM (PARTIAL), EEG-DINO (PARTIAL), LuMamba (PARTIAL), BrainWave (EEG + iEEG) (PARTIAL) |
| TUEV | BIOT [H], LaBraM [P], EEGPT (Wang et al.) [H], CBraMod [P], NeuroLM [P], REVE [P], CSBrain [P], CodeBrain [P], BrainOmni [H], Uni-NTFM [P], EEG-DINO [P] | BENDR (PARTIAL), LaBraM (PARTIAL), CBraMod (PARTIAL), NeuroLM (PARTIAL), FEMBA (PARTIAL), LUNA (PARTIAL), REVE (PARTIAL), CSBrain (PARTIAL), EEGFormer (PARTIAL), Neuro-GPT (PARTIAL), CodeBrain (PARTIAL), Uni-NTFM (PARTIAL), EEG-DINO (PARTIAL), LuMamba (PARTIAL), BrainWave (EEG + iEEG) (PARTIAL) |
| BCIC-IV-2a | BENDR [H], EEGPT (Wang et al.) [H], CBraMod [H], REVE [H], CSBrain [H], Neuro-GPT [H], Uni-NTFM [H], LCM (Large Cognition Model) [H] | - |
| SEED-V | LaBraM [H], CBraMod [H], LUNA [H], CSBrain [H], CodeBrain [H], Uni-NTFM [I], EEG-DINO [H], EEGPT/BrainGPT (Yue et al.) [?] | LaBraM (RELATED), NeuroLM (IN), Uni-NTFM (IN) |
| TUSL | NeuroLM [P], FEMBA [P], LUNA [P], CSBrain [P], EEGFormer [P], Uni-NTFM [P], LuMamba [P] | BENDR (PARTIAL), LaBraM (IN), CBraMod (PARTIAL), NeuroLM (PARTIAL), FEMBA (PARTIAL), LUNA (PARTIAL), REVE (PARTIAL), CSBrain (PARTIAL), EEGFormer (PARTIAL), Neuro-GPT (PARTIAL), CodeBrain (PARTIAL), Uni-NTFM (PARTIAL), EEG-DINO (PARTIAL), LuMamba (PARTIAL), BrainWave (EEG + iEEG) (PARTIAL) |
| FACED | CBraMod [H], REVE [H], CSBrain [H], CodeBrain [H], BrainOmni [H], EEGPT/BrainGPT (Yue et al.) [?] | - |
| MentalArithmetic | CBraMod [H], NeuroLM [H], REVE [H], CSBrain [H], CodeBrain [H], Uni-NTFM [H] | - |
| CHB-MIT | BIOT [H], CBraMod [H], CSBrain [H], CodeBrain [H], BrainWave (EEG + iEEG) [H] | - |
| EEGMMIDB | BENDR [H], CBraMod [H], REVE [H], CSBrain [H], BrainOmni [H] | LaBraM (IN), EEGPT (Wang et al.) (IN), NeuroLM (IN), LCM (Large Cognition Model) (IN) |
| HMC | NeuroLM [H], REVE [H], CSBrain [H], Uni-NTFM [H], EEGPT/BrainGPT (Yue et al.) [?] | BrainWave (EEG + iEEG) (IN) |
| ISRUC | CBraMod [H], REVE [H], CSBrain [H], CodeBrain [H] | - |
| Mumtaz2016 | CBraMod [H], REVE [H], CSBrain [H], BrainOmni [H], BrainWave (EEG + iEEG) [H] | - |
| BCIC2020-3 | CBraMod [H], REVE [H], CSBrain [H], CodeBrain [H] | - |
| SHU-MI | CBraMod [H], CSBrain [H], CodeBrain [H], BrainOmni [H] | - |
| TUAR | FEMBA [P], LUNA [P], EEGFormer [P], LuMamba [P] | BENDR (PARTIAL), LaBraM (IN), CBraMod (PARTIAL), NeuroLM (PARTIAL), FEMBA (PARTIAL), LUNA (PARTIAL), REVE (PARTIAL), CSBrain (PARTIAL), EEGFormer (PARTIAL), Neuro-GPT (PARTIAL), CodeBrain (PARTIAL), Uni-NTFM (PARTIAL), EEG-DINO (PARTIAL), LuMamba (PARTIAL), BrainWave (EEG + iEEG) (PARTIAL) |
| ADFTD | BrainOmni [H], Uni-NTFM [H], BrainWave (EEG + iEEG) [H] | - |
| Sleep-EDFx | BENDR [H], EEGPT (Wang et al.) [H], EEGPT/BrainGPT (Yue et al.) [?] | BrainWave (EEG + iEEG) (IN) |
| BCIC-IV-2b | EEGPT (Wang et al.) [H], LCM (Large Cognition Model) [H] | REVE (IN) |
| KaggleERN | BENDR [H], EEGPT (Wang et al.) [H] | LaBraM (IN), NeuroLM (IN) |
| PhysioNet-P300 | BENDR [H], EEGPT (Wang et al.) [H] | - |
| SEED-VIG | CBraMod [H], CSBrain [H] | - |
| TDBrain | Uni-NTFM [H], LuMamba [H] | REVE (IN) |
| ADHD-Adult | BrainWave (EEG + iEEG) [H] | - |
| ADHD-Child | BrainWave (EEG + iEEG) [H] | - |
| APAVA | LuMamba [H] | - |
| Absence-16 | BrainWave (EEG + iEEG) [?] | - |
| Atonic-5 | BrainWave (EEG + iEEG) [?] | - |
| BCIC-IV-1 | EEGPT/BrainGPT (Yue et al.) [?] | LaBraM (IN), NeuroLM (IN), Uni-NTFM (IN) |
| Clonic-6 | BrainWave (EEG + iEEG) [?] | - |
| DEAP | EEGPT/BrainGPT (Yue et al.) [?] | - |
| DRE-Clinical | BrainWave (EEG + iEEG) [?] | - |
| Depression-122 | BrainWave (EEG + iEEG) [H] | - |
| EEGMat | EEGPT/BrainGPT (Yue et al.) [?] | - |
| FNUSA-iEEG | BrainWave (EEG + iEEG) [H] | - |
| IIIC | BIOT [H] | - |
| IMG | EEGPT/BrainGPT (Yue et al.) [?] | - |
| MIBCI | EEGPT/BrainGPT (Yue et al.) [?] | - |
| Mayo-iEEG | BrainWave (EEG + iEEG) [H] | - |
| MoBI | LaBraM [H] | - |
| Neonate | EEGFormer [H] | - |
| PD31 | BrainOmni [H] | REVE (IN), BrainWave (EEG + iEEG) (IN) |
| SD-71 | BrainWave (EEG + iEEG) [H] | - |
| SEED | NeuroLM [?] | LaBraM (IN), EEGPT (Wang et al.) (IN), NeuroLM (UNCLEAR), LCM (Large Cognition Model) (IN) |
| SEED-IV | EEGPT/BrainGPT (Yue et al.) [?] | LaBraM (IN), NeuroLM (IN), Uni-NTFM (IN) |
| SPE | EEGPT/BrainGPT (Yue et al.) [?] | - |
| STEW | EEGPT/BrainGPT (Yue et al.) [?] | - |
| Schizophrenia-28 | BrainWave (EEG + iEEG) [H] | - |
| Siena | CSBrain [H] | LaBraM (IN), NeuroLM (IN), LUNA (IN), REVE (IN), Uni-NTFM (IN), BrainWave (EEG + iEEG) (IN) |
| SomatoMotor | BrainOmni [H] | - |
| TUSZ | EEGFormer [P] | BENDR (PARTIAL), LaBraM (IN), CBraMod (PARTIAL), NeuroLM (PARTIAL), FEMBA (PARTIAL), LUNA (PARTIAL), REVE (PARTIAL), CSBrain (PARTIAL), EEGFormer (PARTIAL), Neuro-GPT (PARTIAL), CodeBrain (PARTIAL), Uni-NTFM (PARTIAL), EEG-DINO (PARTIAL), LuMamba (PARTIAL), BrainWave (EEG + iEEG) (PARTIAL) |

PARTIAL in the right-hand column means the model pretrained on TUEG or a TUEG sub-corpus and this set is a TUH derivative. `RELATED` = same-family sibling only.

## 5. Datasets that are HELD-OUT for most or all models (candidates for a clean re-evaluation)

**A. Evaluated by 2+ audited models, HELD-OUT in every case where the label is known** (`[?]` = a model whose pretraining list is missing):

| Dataset | # models | Evaluated by | Note: inside pretraining of |
|---|---|---|---|
| BCIC-IV-2a | 8 | BENDR [H], EEGPT (Wang et al.) [H], CBraMod [H], REVE [H], CSBrain [H], Neuro-GPT [H], Uni-NTFM [H], LCM (Large Cognition Model) [H] | - |
| FACED | 6 | CBraMod [H], REVE [H], CSBrain [H], CodeBrain [H], BrainOmni [H], EEGPT/BrainGPT (Yue et al.) [?] | - |
| MentalArithmetic | 6 | CBraMod [H], NeuroLM [H], REVE [H], CSBrain [H], CodeBrain [H], Uni-NTFM [H] | - |
| CHB-MIT | 5 | BIOT [H], CBraMod [H], CSBrain [H], CodeBrain [H], BrainWave (EEG + iEEG) [H] | - |
| EEGMMIDB | 5 | BENDR [H], CBraMod [H], REVE [H], CSBrain [H], BrainOmni [H] | LaBraM (IN), EEGPT (Wang et al.) (IN), NeuroLM (IN), LCM (Large Cognition Model) (IN) |
| HMC | 5 | NeuroLM [H], REVE [H], CSBrain [H], Uni-NTFM [H], EEGPT/BrainGPT (Yue et al.) [?] | BrainWave (EEG + iEEG) (IN) |
| ISRUC | 5 | CBraMod [H], REVE [H], CSBrain [H], CodeBrain [H] | - |
| Mumtaz2016 | 5 | CBraMod [H], REVE [H], CSBrain [H], BrainOmni [H], BrainWave (EEG + iEEG) [H] | - |
| BCIC2020-3 | 4 | CBraMod [H], REVE [H], CSBrain [H], CodeBrain [H] | - |
| SHU-MI | 4 | CBraMod [H], CSBrain [H], CodeBrain [H], BrainOmni [H] | - |
| ADFTD | 3 | BrainOmni [H], Uni-NTFM [H], BrainWave (EEG + iEEG) [H] | - |
| Sleep-EDFx | 3 | BENDR [H], EEGPT (Wang et al.) [H], EEGPT/BrainGPT (Yue et al.) [?] | BrainWave (EEG + iEEG) (IN) |
| BCIC-IV-2b | 2 | EEGPT (Wang et al.) [H], LCM (Large Cognition Model) [H] | REVE (IN) |
| KaggleERN | 2 | BENDR [H], EEGPT (Wang et al.) [H] | LaBraM (IN), NeuroLM (IN) |
| PhysioNet-P300 | 2 | BENDR [H], EEGPT (Wang et al.) [H] | - |
| SEED-VIG | 2 | CBraMod [H], CSBrain [H] | - |
| TDBrain | 2 | Uni-NTFM [H], LuMamba [H] | REVE (IN) |

**B. Appear in no audited model's pretraining list at all** (a new model can use these without checking a competitor's corpus, subject to the caveats below):

ADFTD, ADHD-Adult, ADHD-Child, APAVA, Absence-16, Atonic-5, BCIC-IV-2a, BCIC2020-3, CHB-MIT, Clonic-6, DEAP, DRE-Clinical, Depression-122, EEGMat, FACED, FNUSA-iEEG, IIIC, IMG, ISRUC, MIBCI, Mayo-iEEG, MentalArithmetic, MoBI, Mumtaz2016, Neonate, PhysioNet-P300, SD-71, SEED-VIG, SHU-MI, SPE, STEW, Schizophrenia-28, SomatoMotor

**C. Datasets that ARE inside some audited model's pretraining (avoid when comparing against that checkpoint):** SEED-V -> LaBraM (RELATED), NeuroLM (IN), Uni-NTFM (IN); EEGMMIDB -> LaBraM (IN), EEGPT (Wang et al.) (IN), NeuroLM (IN), LCM (Large Cognition Model) (IN); HMC -> BrainWave (EEG + iEEG) (IN); Sleep-EDFx -> BrainWave (EEG + iEEG) (IN); BCIC-IV-2b -> REVE (IN); KaggleERN -> LaBraM (IN), NeuroLM (IN); TDBrain -> REVE (IN); BCIC-IV-1 -> LaBraM (IN), NeuroLM (IN), Uni-NTFM (IN); PD31 -> REVE (IN), BrainWave (EEG + iEEG) (IN); SEED -> LaBraM (IN), EEGPT (Wang et al.) (IN), NeuroLM (UNCLEAR), LCM (Large Cognition Model) (IN); SEED-IV -> LaBraM (IN), NeuroLM (IN), Uni-NTFM (IN); Siena -> LaBraM (IN), NeuroLM (IN), LUNA (IN), REVE (IN), Uni-NTFM (IN), BrainWave (EEG + iEEG) (IN)

The five TUH sets (TUAB, TUEV, TUSL, TUAR, TUSZ) are inside the TUEG-based pretraining of 15 audited models. Of the papers that evaluate on them, only FEMBA, LUNA and LuMamba claim subject-level removal; CBraMod and CodeBrain give control runs; LaBraM and REVE claim recording-level removal.

**Caveats on list B.** (1) LaBraM and NeuroLM include 342 h of un-itemised self-collected EEG; BIOT and BrainWave include private corpora; REVE lists 56 OpenNeuro sets by accession number only. Overlaps with these cannot be excluded from the papers. (2) BrainGPT/EEGPT-Yue does not list its pretraining data. (3) Dataset identity is matched by name and citation; sibling releases (e.g. SEED vs SEED-V) are not merged. (4) Cross-checks of REVE's OpenNeuro IDs showed ds002778 (UCSD Parkinson's resting EEG) is the same data as BrainOmni's PD31 and BrainWave's PD-31.

## 6. Which headline results are in-corpus, paper by paper

- **BENDR**: No in-corpus headline results.
- **BIOT**: Unsupervised-pretrained headline numbers are clean. The 6-EEG-dataset variant uses same-dataset train splits.
- **LaBraM**: Tables 1-2 (TUAB, TUEV) are PARTIAL: same-institution TUH sub-corpora (TUAR, TUEP, TUSL, TUSZ) are in the pretraining set; authors claim recording-level disjointness.
- **EEGPT (Wang et al.)**: All headline results are HELD-OUT.
- **CBraMod**: 2 of 13 reported datasets (TUEV, TUAB) are PARTIAL; both have an exclusion control with small deltas.
- **NeuroLM**: 3 of 6 headline tasks (TUAB, TUEV, TUSL) are PARTIAL with no control; SEED is UNCLEAR.
- **FEMBA**: All 3 reported datasets are TUH derivatives (PARTIAL), with a subject-level exclusion claim.
- **LUNA**: 3 of 4 reported datasets are TUH derivatives (PARTIAL) with a strong exclusion claim; SEED-V is HELD-OUT.
- **REVE**: 2 of 10 downstream tasks (TUEV, TUAB) are PARTIAL; recording-level removal claimed, no control.
- **CSBrain**: 3 of 16 datasets (TUEV, TUAB, TUSL) are PARTIAL with no control.
- **EEGFormer**: 4 of 5 reported datasets are TUH derivatives (PARTIAL); only Neonate is HELD-OUT.
- **Neuro-GPT**: Clean.
- **CodeBrain**: 2 of 10 (TUEV, TUAB) PARTIAL; both have an exclusion control; they sit in the appendix.
- **BrainOmni**: All headline results are HELD-OUT.
- **Uni-NTFM**: 4 of 9 downstream tasks (TUAB, TUEV, TUSL, SEED) overlap the pretraining corpus, none with a control.
- **EEG-DINO**: 2 of 3 datasets PARTIAL, no control.
- **LuMamba**: 3 of 5 PARTIAL with an exclusion claim.
- **LCM (Large Cognition Model)**: Clean.
- **BrainWave (EEG + iEEG)**: No public downstream set is in-corpus; 4 private sets are UNCLEAR.
- **EEGPT/BrainGPT (Yue et al.)**: Cannot be assessed (pretraining list missing).

## 7. Other issues seen while reading (not contamination of the pretraining kind)

- BIOT's '6 EEG datasets' variant and NeuroLM's multi-task instruction tuning use labeled train splits of the downstream tasks; test splits stay separate.
- LUNA (TUAR, TUSL) and LuMamba (TUSL, TUAR) fine-tune with sample-level random 80/10/10 splits, so subjects can appear in both train and test inside the downstream task. LUNA states this on p.6.
- EEGPT (Wang et al.) text in the introduction names three pretraining sets while Table 1 lists five.
- NeuroLM's SEED row: original SEED is downstream but only SEED-IV/V/GER/FRA are itemised; the hours (170.54 h) differ from LaBraM's list (166.75 h, which names SEED).
- Uni-NTFM's 'SEED' downstream spec (16 subjects, 5 classes) matches SEED-V, which is in its pretraining list; labelled IN-CORPUS by inference from Table 7.

## 8. Not audited

- Gram (ICASSP 2025): No arXiv/open full text found by web search; only an AMiner listing. Not audited.
- BrainBERT, Brant, Brant-X: Intracranial/physiological-signal models, outside the scalp-EEG scope.
- EEGMamba (2407.20254): Multi-task classifier, not a pretrain-then-finetune foundation model; not audited.

## Additions after 2026-10-25 (not included in the 20-model counts above)

| Model | Pretraining corpora (own list) | Downstream results, mark per dataset | I / P / H / ? |
|---|---|---|---|
| **BrainCodes** (arXiv 2610.29320, 21 Oct 2026) | Encoder: HBN-EEG releases R3, R4, R7, R8 (about 820 h, 1,131 participants), §3.1. Tokenizer: frozen BioCodec (arXiv 2510.09095), itself pretrained on the TUH-EEG corpus with TUAB and TUEV excluded (BioCodec §3.1) | TUAB [P*], TUEV [P*], ISRUC [H], Siena [H], PhysioNet-MI [H], BCI IV-2a [H], EEGMAT [H], High-Gamma [H], OpenBMI-ERP [H], Pavlov-Memory [H], PhysioNet-Eyes [H], EESM23 [H], Muse-Motor [H], Muse-Mental [H] | 0 / 2 / 12 / 0 |

**Overlap through a frozen component.** BrainCodes' own encoder never sees TUH data, but its frozen tokenizer was pretrained on TUH-EEG. TUAB and TUEV are TUH sub-corpora, so their results are marked [P*]: same collection, with a stated exclusion (BioCodec: "we explicitly exclude two subsets designated for downstream evaluation"). Overlap checks should cover every pretrained component, not only the encoder being fine-tuned. Pending author verification.
