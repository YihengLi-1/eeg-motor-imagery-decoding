# PyTorch-MPS grouped-convolution bug and its effect on the from-scratch baselines (found 2026-11-01)

## What was wrong
Every from-scratch **EEGNet** row in the matrices (PhysioNet `step3c_fullmatrix_results.json`, Cho2017
`heldout/results_cho2017_matrix.json`, both bands, 4 seeds each) had training-set accuracy 0.49–0.53: the network
never fitted even the training data, while the same loop fitted the pretrained LaBraM/CBraMod to 0.95–1.0.
EEGNet trained in isolation on CPU learns normally (train acc 0.53 → 0.72 over 5 epochs on 10 Cho2017 subjects),
the identical script on MPS stays at 0.50.

## Root cause
On torch 2.14.1 / macOS, `F.conv2d` with `groups > 1` on the MPS device returns **wrong forward and backward
results whenever the batch size is a multiple of 32**. Verified standalone with the EEGNet depthwise conv
(8→16, kernel (64,1), groups=8):

| batch | time points | result vs CPU |
|---|---|---|
| 16, 31, 33 | 1001 | identical |
| 32, 64 | 1001 | wrong (max abs diff ≈ 3 on outputs of scale ≈ 26; weight-grad rel diff 1.0) |
| 32, 64, 128, 256 | 500 / 250 / 125 | wrong |
| 16 / 8 | 2002 / 4004 (same numel as 32×1001) | identical |

So the trigger is the batch dimension, not tensor size. Ungrouped `Conv2d` is correct at every batch size, and
LaBraM / CBraMod forward passes agree with CPU at batch 32 (no grouped conv), so the pretrained rows are unaffected.
Our EEGNet / probe / randFT recipe uses batch 32 → EEGNet broken; fullFT recipes use batch 16 → fine.

## Upstream status (checked 2026-11-01)
Already reported as [pytorch/pytorch#202452](https://github.com/pytorch/pytorch/issues/202452) (opened 2026-10-28, high priority,
"Conv2d with a tall filter (height 64-255) silently returns wrong results at some batch sizes"); maintainer Isalia20 is
migrating conv2d from MPSGraph to a Metal kernel. #188335 (closed 2026-07-06) fixed the same defect only for filter
height >= 256. What that report does not say and we can add: the backward pass is wrong too (weight- and input-gradient
relative error about 1.0), and the (1,16) separable depthwise conv is unaffected, so it is the tall filter, not `groups>1`
per se. Repro + output + draft comment: `docs/pytorch_issue/`. Posted 2026-11-01: https://github.com/pytorch/pytorch/issues/202452#issuecomment-6569066843

## Fix
`heldout/mps_groupconv.py` patches `torch.nn.functional.conv1d/conv2d` to run grouped convolutions on MPS in two
batch halves when `batch % 32 == 0`; it self-checks against CPU at import and raises if the patched op disagrees.
Imported by `step3d_matrix.py` and `heldout/live_heldout.py` next to `mps_sdpa`. End-to-end check: EEGNet at
batch 32, patched MPS vs CPU: forward max diff 1.5e-7, no parameter with a gradient mismatch.

## What has to be rerun
- EEGNet, both datasets, both bands, seeds 0–3 (rows removed from the result files; backup in `heldout/_backup_20261101_0011/`).
- `heldout/edge_quant_results.json` EEGNet rows (4 seeds) were produced the same way and are suspect; rerun before quoting.
- `randFT` (LaBraM random init) seed 0 was also removed: it collapsed to a constant prediction, but that is **not**
  this bug (LaBraM has no grouped conv; CPU runs collapse too). Its recipe is still open.
- Any number for EEGNet in `docs/heldout_leaderboard.md`, `docs/pretraining_overlap_eeg_fm.md` and the paper draft is
  stale until the rerun finishes.

## Lesson (gate)
A from-scratch baseline that does not fit its own training set is a pipeline bug until proven otherwise; the
summary now prints training-set accuracy next to test accuracy so this is visible, and `mps_groupconv.self_check()`
runs on every import.

## Inventory of affected results (2026-11-01 01:30, before any rerun except Cho2017 EEGNet)
Every script trained or evaluated EEGNet on MPS with batch 32 or 64 (`predict()` also uses batch 32, so even
inference of a correct model was wrong except for the final partial batch). TimeMIL uses grouped `conv1d`
(wavelet kernels, `groups=D`) with batch 32 and is affected too (it still reached train acc 0.7–0.83, so the damage
is smaller than for EEGNet, but its numbers are not trustworthy).

| Result | Script | Batch | Status |
|---|---|---|---|
| Cho2017 EEGNet (2 bands × 4 seeds) | `step3d_matrix.py` | 32 | **rerun done** 01:16: 70.7 ± 2.6 (0.1–75 Hz), 66.6 ± 1.3 (8–30 Hz); was 52.3 / ~50 |
| PhysioNet EEGNet (2 bands × 4 seeds) | `step3d_matrix.py` | 32 | **rerun done** 01:28: 82.2 ± 1.5 (0.1–75 Hz), 50.2 ± 3.6 (8–30 Hz); was 54.9 / 49.8 |
| README headline / Split A (LaBraM vs EEGNet) | `step3_compare.py`, `step3b_broadband.py`, `step3c_fullmatrix.py` | 32 | superseded by the matrix; do not quote |
| Cho2017 TimeMIL (4 seeds, 59.2 ± 1.6) | `heldout/timemil_heldout.py` | 32 | patched; `run_queue_1101b.sh` forgot `MI_DATASET=cho2017` and produced the PhysioNet row instead (64.2 ± 1.7, new, correct); Cho2017 **rerun done** 10:19 (`run_timemil_cho_1101.sh`): 58.8 ± 3.3 (56.0/59.2/55.8/64.1, train acc 0.70–0.81), was 59.2 ± 1.6 |
| Live bench ds008446 (all 5 rows) | `heldout/live_heldout.py` | 32 | run entirely after the fix (01 Nov 03:40–07:30); nothing to redo |
| Siena / CHB-MIT EEGNet (false alarms, F1), distillation, ensemble | `seizure/szcore_siena.py`, `seizure/distill.py`, `seizure/ensemble.py` | 64 | **rerun done** 2026-11-01 12:30–15:25 (`seizure/run_rerun_1101.sh`, log `seizure/progress_rerun_1101.log`; caches had to be rebuilt first). EEGNet FA/24h: Siena k7 27.6 (was 29.6), Siena LOSO 24.2 (was 26.4), CHB-MIT 18.3 (was 12.4); pooled F1 0.18 / 0.18 / 0.14 (was 0.24 / 0.16 / 0.17); sensitivity fell (k7 0.43, was 0.64). KD 36.8, KDbg30 44.1 FA/24h. Ensemble k7 F1 0.43 FA 7.0 (was 0.37 / 10.9); LOSO and CHB-MIT ensembles still do not help. LaBraM rows untouched. Old results and weights in `seizure/_backup_20261101_prefix_mps/` |
| Edge quantisation EEGNet rows | `heldout/edge_quant.py` | 32 | **rerun done** 15:24: fp32 70.6 ± 2.5, int8 dynamic 70.4 ± 3.1, int8 static 65.2 ± 1.7 (was 59.0 / 58.8 / 58.3); fullFT rows unchanged (65.8 / 64.4 / 56.3). Not quoted in any public document yet |
| LaBraM / CBraMod probe & fine-tune, Riemannian, NeuroRVQ/other FM rows | various | 16/32 | not affected (no grouped conv; LaBraM/CBraMod forward verified equal to CPU at batch 32) |

Paper (`paper_tmlr/main.tex`) claims that rest on EEGNet and must be rewritten after the reruns: the 24.2 / 13.5-point
motor-imagery margins, the "advantage falls from in-corpus to out-of-corpus" framing (on Cho2017 the corrected EEGNet,
70.7 %, is now above fine-tuned LaBraM, 65.8 %), and the Siena vs CHB-MIT false-alarm comparison.

## Corrected motor-imagery picture (2026-11-01 01:28, broadband 0.1–75 Hz, 4 subject-level splits)
| | PhysioNet (in LaBraM corpus) | Cho2017 (held out) | drop |
|---|---|---|---|
| EEGNet from scratch | 82.2 ± 1.5 | 70.7 ± 2.6 | −11.5 |
| CBraMod fine-tune (neither dataset in corpus) | 82.3 ± 1.1 | 71.4 ± 3.3 | −10.9 |
| LaBraM fine-tune | 79.1 ± 2.0 | 65.8 ± 2.6 | −13.3 |
| LaBraM linear probe | 48.5 ± 2.7 | 51.7 ± 0.6 | – |

Fine-tuned LaBraM is below a from-scratch EEGNet on both datasets; CBraMod ties EEGNet. The in-corpus → held-out
drop is the same size for a model with no pretraining at all, so it measures dataset difficulty, not overlap. The
paper's "pretraining advantage shrinks out of corpus" framing has to go; the overlap-does-not-explain-the-gap
conclusion survives in a stronger form (there is no gap to explain).

## Same-architecture random-init control (2026-11-01 03:38, Cho2017, broadband, 4 splits)
CBraMod architecture trained from scratch: 62.4 ± 3.1 (per seed 58.8 / 62.8 / 61.8 / 66.3; train acc 0.88).
Pretrained CBraMod fine-tuned: 71.4 ± 3.3 → pretraining is worth about 9 points for this architecture, which is the
correct ablation for "does the pretraining help" (the EEGNet comparison answers a different question: is the
pretrained model better than a small purpose-built net; here it is not). LaBraM's random-init control still has no
working recipe (collapses to a constant prediction on CPU and MPS; several learning rates, head init scales, warmup,
AdamW and gradient clipping tried on 2026-10-31/11-01).

## Second defect in the fix (found 2026-11-03)
The 11-01 patch split a batch whose size is a multiple of 32 into two halves once. For batch 64 and 128 the halves
(32, 64) are still multiples of 32, so the result stayed wrong: EEGNet forward vs CPU, max abs diff 0.23 at batch 64 and
0.22 at batch 128 (correct at 16, 31, 32, 48, 50, 96). The self-check only tested batch 32, so it passed.
Fix: the wrapper now splits recursively (`heldout/mps_groupconv.py`), and `self_check()` tests batch 32, 64 and 128.
After the fix all of 32/64/128/256 agree with CPU (max diff ≤ 6e-8).

Affected: everything that trained or evaluated EEGNet on MPS with batch 64 after 11-01, i.e. the seizure pipeline
(`seizure/szcore_siena.py` trains EEGNet with batch 64 and computes probability traces with batch 64; `seizure/distill.py`
trains the KD students with batch 64). The 11-01 seizure "rerun done" EEGNet / KD / KDbg30 / ensemble numbers are
therefore still invalid. Rows backed up to `seizure/_backup_20261103_prerecursive/` (results JSONs, models, the 11-01 log)
and removed; rerun with `seizure/run_rerun_1103.sh` (log `seizure/progress_rerun_1103.log`). Motor-imagery results
(`step3d_matrix.py`, batch 32 for EEGNet) and the live bench / edge quantisation (batch 32) are not affected.
