# BraTS 2024 GLI — Results Summary (Presentation)

Brain tumor segmentation (TC/WT/ET) with Monte Carlo Dropout uncertainty, on the BraTS 2024 GLI
(post-treatment glioma) dataset. Two model generations were trained; this session added (1) MC
Dropout uncertainty maps + calibration for the current best model, and (2) a real MC-dropout
comparison against a published, downloaded model — not just a literature Dice table.

This folder (`v2/results_summary/`) is a self-contained snapshot: every figure/table referenced below
is a copy stored right next to this file, so the whole folder can be shared as-is. Source scripts
still write their canonical outputs to `v2/uncertainty_vis_v2_1/`, `v2/baseline_comparison/`, etc. —
this folder is a presentation-friendly copy, not the reproducible source of truth.

---

## 1. Project Trajectory — v1 → v2.1

| | v1 — plain 3D U-Net | v2.1 — Attention U-Net (current best) |
|---|---|---|
| Dice TC / WT / ET / Mean | 0.8638 / 0.9087 / 0.8503 / 0.8743 | **0.8791 / 0.9206 / 0.8613 / 0.8870** |
| HD95 TC / WT / ET / Mean (mm) | 5.65 / 6.11 / 5.74 / 5.83 | **4.83 / 4.87 / 5.06 / 4.92** |
| Eval set | own 80/20 holdout, reshuffled split | same 324-subject **persistent** val split |

v2.1 improves on every metric (+1.3 pts mean Dice, ~16% lower mean HD95). v1 used a reshuffle-by-seed
val split; v2.1 introduced a persistent split so results are reproducible across resumes/reruns.

- Details: `../../PROGRESS.md` (§1–13), `../V2_1_ARCHITECTURE.md`

---

## 2. MC Dropout Uncertainty Maps

### v1 (plain U-Net)

| Region | Mean Entropy |
|---|---|
| TP | 0.039 |
| FP | 0.236 |
| FN | ~0.20 |

FP entropy ≈ **6× TP entropy** — the model flags its own mistakes.
Script: `../../visualize_uncertainty.py` (output not currently regenerated on disk — gitignored,
reproducible by rerunning the script; numbers quoted from `../../ANALYSIS.md` §6b).

### v2.1 (Attention U-Net) — new this session

| Region | Mean Entropy |
|---|---|
| TN | 0.0002 |
| TP | 0.0285 |
| FP | 0.1960 |
| FN | 0.1112 |

FP entropy ≈ **6.9× TP entropy** — same behavior holds after the attention-gate upgrade.

**New: voxel-level calibration** (50 val subjects, brain-mask restricted):
- **Expected Calibration Error (ECE) = 0.0080** — softmax confidence is well calibrated
- **AUROC = 0.8956** — per-voxel entropy alone strongly predicts misclassified voxels

**Figures (in this folder):**
- `examples/example_01..10_<subject>.png` — 10 individual paper-style figures, one per subject
  (T1c / GT / MC-Pred / Entropy / Overlay panels), black background
- `uncertainty_stats.png` — entropy violin + bar chart by region (TN/TP/FP/FN)
- `calibration.png` — reliability diagram (ECE) + ROC curve (AUROC)
- `uncertainty_summary.json` — all numbers above, machine-readable

(Canonical source: `../uncertainty_vis_v2_1/`, produced by `../visualize_uncertainty_v2.py`)

---

## 3. Comparison to Other Models

### 3a. Literature comparison (Dice only, existing — `../../ANALYSIS.md` §7)

Numbers taken from published papers, on their own official/leaderboard eval sets (not ours) —
context only, not head-to-head.

| Method | Dice TC | Dice WT | Dice ET | Eval set | Dataset |
|---|---|---|---|---|---|
| **Ours — v1 (3D U-Net + MC Dropout)** | 0.8638 | 0.9087 | 0.8503 | Internal holdout | BraTS 2024 (post-tx) |
| BraTS 2024 challenge submission (Court et al., arXiv 2405.18368) | 0.7499 | 0.9055 | 0.8124 | Official validation, lesion-wise | BraTS 2024 (post-tx) |
| BraTS 2023 winner (arXiv 2402.17317) | 0.8673 | 0.9005 | 0.8509 | Official val, lesion-wise | BraTS 2023 (pre-tx) |
| nnUNetFormer (Gao et al.) | 0.921 | 0.936 | 0.872 | Scan-level | BraTS 2021 (pre-tx) |
| SwinUNETR (Hatamizadeh et al.) | 0.831 | 0.852 | 0.799 | Scan-level | BraTS 2021 (pre-tx) |

Full caveats (post- vs pre-treatment mismatch, lesion-wise vs scan-level metric mismatch): `../../ANALYSIS.md` §7.

### 3b. Published-model MC Dropout comparison — new this session (`../BASELINE_COMPARISON.md`)

Downloaded **MONAI's `brats_mri_segmentation`** bundle (SegResNet, Myronenko 2018, trained on BraTS
2018) — the only publicly downloadable pretrained BraTS checkpoint found with genuine trained dropout
(`dropout_prob=0.2`; SwinUNETR/nnU-Net BraTS21 alternatives have no usable dropout). Ran it through an
adapter pipeline (channel reorder, normalization, sliding-window inference, MC Dropout) on **our own
identical 324-subject val split** — a true head-to-head, unlike 3a.

**Dice / HD95:**

| Model — Mode | Dice TC | Dice WT | Dice ET | Dice Mean | HD95 Mean (mm) |
|---|---|---|---|---|---|
| **v2.1 (ours) — Deterministic** | **0.8791** | **0.9206** | **0.8613** | **0.8870** | **4.92** |
| v2.1 (ours) — MC Dropout (20) | 0.8791 | 0.9205 | 0.8612 | 0.8869 | 4.92 |
| SegResNet (published, BraTS 2018) — Deterministic | 0.5620 | 0.7797 | 0.3955 | 0.5791 | 18.09 |
| SegResNet (published, BraTS 2018) — MC Dropout (20) | 0.5723 | 0.7896 | 0.3961 | 0.5860 | 17.13 |

The published baseline scores much lower on our data (0.58 vs 0.887 mean Dice) — **expected domain
shift**: it has never seen BraTS 2024's post-treatment scans (surgical cavities, radiation effects,
pseudoprogression). ET is hit hardest (0.396 vs 0.861); WT degrades least (0.780 vs 0.921).

**MC Dropout uncertainty, ours vs. the baseline (whole-tumor / WT-channel entropy):**

| | TN | TP | FP | FN |
|---|---|---|---|---|
| Ours (v2.1) | 0.0002 | 0.0285 | 0.1960 | 0.1112 |
| SegResNet baseline | 0.0136 | 0.1725 | 0.4507 | 0.3656 |

Both show correct qualitative ordering (TN < TP < FN ≤ FP), but ours separates correct/incorrect
predictions far more sharply — **FP/TP ratio 6.9× (ours) vs. 2.6× (baseline)**. Takeaway: MC-dropout
uncertainty transfers qualitatively across domains, but is much blunter on a model never trained on
the target data — a model actually trained on your domain gives you a sharper uncertainty signal, not
just better raw accuracy.

**Figures/data (in this folder):**
- `comparison_table.png` — Dice/HD95, v2.1 vs. baseline, det + MC-20
- `entropy_comparison.png` — entropy bar chart, ours vs. baseline
- `segresnet_results.json` — full per-region (TC/WT/ET) entropy stats
- `v21_det.json`, `v21_mc20.json` — our own numbers, regenerated

(Canonical source: `../baseline_comparison/`, produced by `../run_baseline_segresnet.py` +
`../compare_baseline.py`. Full methodology + caveats: `../BASELINE_COMPARISON.md`)

---

## 4. All Figures in This Folder

| Figure | File |
|---|---|
| v2.1 example figures (10 subjects, 1 each) | `examples/example_01_<subject>.png` ... `example_10_<subject>.png` |
| v2.1 entropy-by-region stats | `uncertainty_stats.png` |
| v2.1 calibration (ECE + ROC/AUROC) | `calibration.png` |
| v2.1 vs. SegResNet Dice/HD95 comparison table | `comparison_table.png` |
| Ours vs. published-baseline entropy | `entropy_comparison.png` |
| v2.1 deterministic inference metrics table | `inference_metrics_table.png` |
| v2.1 MC-20 inference metrics table | `montecarlo_metrics_table.png` |
| v2.1 MC per-pass stability table | `mc_per_pass_table.png` / `mc_per_pass_table.csv` |

Not copied here (too many files): 324 per-subject inference figures at
`../inference_report_v2_1/slices/*.png` and `../montecarlo_v2_1/slices/*.png`.

## 5. All Source Docs

| Doc | Contents |
|---|---|
| `../../PROGRESS.md` | Full phase-by-phase project log (§1–15) |
| `../../ANALYSIS.md` | v1 deep-dive + literature comparison (§7) |
| `../V2_1_ARCHITECTURE.md` | v2.1 architecture rationale, attention-gate mechanics |
| `../BASELINE_COMPARISON.md` | Published-model MC comparison — full methodology, caveats, results |
