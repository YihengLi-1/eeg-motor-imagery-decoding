# Public EEG foundation-model checkpoints vs. the NeurIPS 2026 Neural Interfaces challenge datasets

Compiled 2026-10-31 by Yiheng Li from the [pretraining-overlap table](pretraining_overlap_eeg_fm.md) (23 models; rows for FEMBA, LUNA, LuMamba, BrainCodes, CodeBrain, Uni-NTFM, Neuro-GPT and NeuroLM confirmed by their authors). "In pretraining of" means the dataset is listed in that model's own stated pretraining corpus; it is not a claim about any competition submission. Datasets with no entry were not found in any itemised corpus, which is not proof of absence: 5 of the 23 models do not itemise their corpora (BIOT, LaBraM, NeuroLM, BrainWave (EEG + iEEG), EEGPT/BrainGPT (Yue et al.)), and REVE lists 56 OpenNeuro sets by accession only.

| Track | Dataset (role in the challenge) | In pretraining of |
|---|---|---|
| Track 01 EEG-to-Image | THINGS-EEG1 / THINGS-EEG2 | none found |
| Track 01 EEG-to-Image | THINGS-MEG (same stimulus set, MEG) | BrainOmni |
| Track 01 EEG-to-Image | Alljoined | none found |
| Track 02 BCI decoding | PROTEUS (NEMAR nm000290) | none found |
| Track 02 BCI decoding | Dreyer 2023 | none found |
| Track 02 BCI decoding | Stieger 2021 | none found |
| Track 02 BCI decoding | Zyma 2019 (Workload) | none found |
| Track 02 BCI decoding | Scherer 2015 | none found |
| Track 03 Sleep onset | Muse Sleep-Onset (NEMAR nm000287) | none found |
| Track 03 Sleep onset | Sleep-EDF | BrainWave (EEG + iEEG), NOMAD-EEG (Large variant only; downstream subjects held out) |
| Track 03 Sleep onset | PhysioNet/CinC Challenge 2018 | none found |
| Track 03 Sleep onset | HMC | BrainWave (EEG + iEEG) |
| Track 03 Sleep onset | SHHS (NSRR, permitted under licence) | BIOT |
| Track 03 Sleep onset | CAP (NSRR) | BrainWave (EEG + iEEG) |
| 2025 challenge data | HBN-EEG | BrainOmni, BrainCodes |

**Why this matters for Rule 04 / Rule 08.** A submission that starts from a public checkpoint inherits that checkpoint's pretraining exposure. A declaration of "external corpora" is only checkable if the checkpoint's own corpus is itemised; for the five models above it is not. The `check_pretraining_overlap` helper proposed in braindecode PR #1493 does this lookup for the models in the table.
