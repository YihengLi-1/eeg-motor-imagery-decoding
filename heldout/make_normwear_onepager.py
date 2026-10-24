"""One-page handout: NormWear on Cho2017 next to EEGNet / LaBraM. Reads the two results files, writes HTML + PDF."""
import json, os, subprocess

ROOT = os.path.expanduser("~/bci-project/heldout")
DST = os.path.expanduser("~/Documents/2027暑研申请/epfl_application/NormWear_Cho2017_onepager")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

base = json.load(open(f"{ROOT}/results_cho2017_matrix.json"))["final_mean_std_over_seeds"]
nw = json.load(open(f"{ROOT}/results_cho2017_normwear.json"))["results"]
bpw = json.load(open(f"{ROOT}/results_cho2017_bandpower.json"))["bandpower_lr"]
pct = lambda m, s: f"{m * 100:.1f}% ± {s * 100:.1f}%"
b = lambda k: pct(base[k]["acc_mean"], base[k]["acc_std"])
n = lambda c, p="avg": pct(nw[f"normwear_probe|{c}|{p}"]["acc_mean"], nw[f"normwear_probe|{c}|{p}"]["acc_std"])
gap = (nw["normwear_probe|nosmooth|avg"]["acc_mean"] - nw["normwear_probe|paper|avg"]["acc_mean"]) * 100

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>NormWear on cross-subject EEG</title>
<style>
@page {{ size: A4; margin: 12mm 15mm; }}
body {{ font-family: "Helvetica Neue", Arial, sans-serif; font-size: 9.6pt; line-height: 1.3; color: #000; margin: 0; }}
h1 {{ font-size: 13pt; margin: 0 0 2px; }} .by {{ font-size: 8.8pt; margin-bottom: 8px; }}
h2 {{ font-size: 10.5pt; margin: 10px 0 3px; border-bottom: 1px solid #000; padding-bottom: 1px; }}
p {{ margin: 0 0 5px; }} ul {{ margin: 0 0 5px; padding-left: 16px; }} li {{ margin: 0 0 2px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9.2pt; margin: 4px 0; }}
th, td {{ border: 1px solid #000; padding: 2px 6px; text-align: right; }} th:first-child, td:first-child {{ text-align: left; }}
.note {{ font-size: 8.4pt; }}
</style></head><body>
<h1>NormWear on multi-channel, cross-subject EEG: a first look (Cho2017 motor imagery)</h1>
<div class="by">Yiheng Li, ASU CS &nbsp;|&nbsp; yihengl4@asu.edu &nbsp;|&nbsp; github.com/YihengLi-1/eeg-motor-imagery-decoding &nbsp;|&nbsp; report: doi.org/10.5281/zenodo.23423535</div>

<p><b>Question.</b> NormWear's EEG evaluation covers 1&ndash;4-channel tasks (Bonn epilepsy, GAMEEMO, DriverFatigue). How do its frozen
features do on a 64-channel motor-imagery dataset where every test subject is unseen, under the exact protocol I used for EEGNet and LaBraM?</p>
<p><b>Setup.</b> Cho2017 (52 subjects, left vs. right hand imagery; not in NormWear's or LaBraM's pretraining corpus). Nine motor channels
(FC3/4, C3/1/z/2/4, CP3/4), &minus;1 to 4 s around the cue. Four random subject splits: 10 test, 5 validation, rest training, the same splits as
the EEGNet/LaBraM rows. NormWear: released checkpoint, frozen encoder, patch-mean embedding per channel concatenated, logistic regression with C
chosen on the validation subjects (as in the paper's linear-probe protocol). Band power: Welch log power in 6 bands (4&ndash;40 Hz), 0 to 4 s, same probe. Accuracy is the mean over test subjects.</p>

<h2>Results (accuracy, mean ± SD over 4 subject splits; chance = 50%)</h2>
<table>
<tr><th>Model</th><th>Adaptation</th><th>Accuracy</th></tr>
<tr><td>EEGNet (64 channels)</td><td>trained from scratch</td><td>{b("EEGNet|broad_0.1-75")}</td></tr>
<tr><td>Log band power, 9 motor channels (classic baseline)</td><td>logistic regression</td><td>{pct(bpw["acc_mean"], bpw["acc_std"])}</td></tr>
<tr><td>LaBraM (64 channels)</td><td>frozen + linear probe</td><td>{b("probe|broad_0.1-75")}</td></tr>
<tr><td>LaBraM (64 channels)</td><td>full fine-tuning</td><td>{b("fullFT|broad_0.1-75")}</td></tr>
<tr><td><b>NormWear, 9 motor channels, paper preprocessing</b></td><td>frozen + logistic regression</td><td><b>{n("paper")}</b></td></tr>
<tr><td>NormWear, same but without the 0.02 s smoothing</td><td>frozen + logistic regression</td><td>{n("nosmooth")}</td></tr>
</table>
<p class="note">[CLS]-token pooling instead of patch mean: {n("paper", "cls")} (paper preprocessing), {n("nosmooth", "cls")} (no smoothing).</p>

<h2>Observations</h2>
<ul>
<li><b>Frozen NormWear features transfer to new subjects.</b> With no motor-imagery data in pretraining and no fine-tuning,
NormWear's frozen embeddings reach {n("paper")}, the same as fully fine-tuned LaBraM and
{(nw["normwear_probe|paper|avg"]["acc_mean"] - bpw["acc_mean"]) * 100:.1f} points above classic log band power on the same 9 channels, while
LaBraM's frozen probe stays near chance. (Input channels differ between rows, as labelled.)</li>
<li><b>The smoothing does not hurt here.</b> The 0.02 s Gaussian smoothing scales 20 Hz by 0.04 (verified numerically), but removing it changes
accuracy by {gap:+.1f} points, so the information NormWear uses for motor imagery survives it.</li>
<li><b>Adaptation gap differs by model.</b> LaBraM's frozen probe and full fine-tuning differ by
{(base["fullFT|broad_0.1-75"]["acc_mean"] - base["probe|broad_0.1-75"]["acc_mean"]) * 100:.1f} points on the same splits, while NormWear's frozen
features already reach that level. Why one frozen representation transfers across subjects and another does not, and whether this can be
predicted before fine-tuning, is the question I would most like to study next.</li>
<li><b>Small code issue.</b> In <code>NormWearModel.get_embedding</code> (and <code>NormWear_API.get_embedding</code>), inputs above 256 Hz go to
<code>torchaudio.Resample</code> as a NumPy array and raise an AttributeError, and inputs at or below 256 Hz other than 65 Hz are not resampled at all.
A 6-line patch makes 512, 200 and 65 Hz inputs all produce the same 65 Hz embedding (submitted as PR #12 on the NormWear repository).</li>
</ul>

<h2>What I could do next</h2>
<ul>
<li>Extend this to CHB-MIT seizure detection (SzCORE event scoring, 980 h) and BCIC-IV-2a, with NormWear fine-tuned as well as probed.</li>
<li>Measure the frozen-vs-fine-tuned gap across EEG foundation models (LaBraM, CBraMod, NormWear) and test whether Fisher-based diagnostics predict it.</li>
<li>Apply my pretraining-overlap audit (20 EEG foundation models, 142 reported results) to wearable PPG/ECG foundation models.</li>
</ul>
<p class="note">Code: heldout/normwear_cho2017.py in the repository above. Preliminary results; single run per split.</p>
</body></html>"""
open(DST + ".html", "w").write(html)
subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={DST}.pdf", DST + ".html"],
               check=True, capture_output=True)
print("wrote", DST + ".pdf", f"(smoothing effect {gap:+.1f} pts)")
