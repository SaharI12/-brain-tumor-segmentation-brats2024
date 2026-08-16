# BraTS 2024 GLI — Results Summary (Presentation)

Brain tumor segmentation (TC/WT/ET) with Monte Carlo Dropout uncertainty, on the BraTS 2024 GLI
(post-treatment glioma) dataset. Two model generations were trained; the paper's results are the
v2.1 Attention U-Net plus its MC Dropout uncertainty maps and calibration analysis.

This folder (`v2/results_summary/`) is a self-contained snapshot: every figure/table referenced below
is a copy stored right next to this file, so the whole folder can be shared as-is. Source scripts
still write their canonical outputs to `v2/uncertainty_vis_v2_1/`, etc. —
this folder is a presentation-friendly copy, not the reproducible source of truth.

---

## 1. Project Trajectory — v1 → v2.1 (accuracy + uncertainty)

| Metric | v1 — Plain 3D U-Net | v2.1 — Attention U-Net (current best) |
|---|---|---|
| Dice TC / WT / ET / Mean | 0.8638 / 0.9087 / 0.8503 / 0.8743 | **0.8791 / 0.9206 / 0.8613 / 0.8870** |
| HD95 TC / WT / ET / Mean (mm) | 5.65 / 6.11 / 5.74 / 5.83 | **4.83 / 4.87 / 5.06 / 4.92** |
| Entropy TN / TP / FP / FN | not computed / 0.039 / 0.236 / ~0.20 | 0.0002 / 0.0285 / 0.1960 / 0.1112 |
| FP/TP entropy ratio | ~6.0× | ~6.9× |
| ECE (calibration) | not computed | 0.0080 |
| AUROC (entropy as error detector) | not computed | 0.8956 |
| Eval set | own 80/20 holdout, reshuffled split | same 324-subject **persistent** val split |

v2.1 improves on every accuracy metric (+1.3 pts mean Dice, ~16% lower mean HD95) and keeps the same
uncertainty behavior — FP entropy stays far above TP entropy, ratio actually rising slightly (6.0× →
6.9×) after the attention-gate upgrade. v1 used a reshuffle-by-seed val split; v2.1 introduced a
persistent split so results are reproducible across resumes/reruns — this means the Dice/HD95 rows are
a clean head-to-head, but v1's entropy numbers (quoted from `ANALYSIS.md` §6b, script output not
regenerated on disk) come from a different held-out sample, and v1 has no TN entropy, ECE, or AUROC
since its uncertainty script predates the calibration analysis added for v2.1.

- Details: `../../PROGRESS.md` (§1–13), `../V2_1_ARCHITECTURE.md`

---

## 2. MC Dropout Uncertainty Maps

Same numbers as the entropy rows in §1's master table — broken out below per version with script/data
provenance.

### v1 (plain U-Net)

| Region | Mean Entropy |
|---|---|
| TP | 0.039 |
| FP | 0.236 |
| FN | ~0.20 |

FP entropy ≈ **6× TP entropy** — the model flags its own mistakes.
(v1 numbers quoted from `../../ANALYSIS.md` §6b; the v1 uncertainty script was removed in the
repo cleanup — the v2.1 pipeline `../visualize_uncertainty_v2.py` supersedes it.)

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

### Literature comparison (Dice only — `../../ANALYSIS.md` §7)

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

**Data files (in this folder):**
- `v21_det.json`, `v21_mc20.json` — our v2.1 numbers (deterministic and MC-20), full per-region detail

---

## 4. All Figures in This Folder

| Figure | File |
|---|---|
| 3D-UNet vs Ours — Dice/HD95 table | `v1_vs_v2_results_table.png` |
| 3D-UNet vs Ours — entropy/ECE/AUROC table | `v1_vs_v2_entropy_table.png` |
| v2.1 example figures (10 subjects, 1 each) | `examples/example_01_<subject>.png` ... `example_10_<subject>.png` |
| v2.1 entropy-by-region stats | `uncertainty_stats.png` |
| v2.1 calibration (ECE + ROC/AUROC) | `calibration.png` |
| v2.1 calibration, tumor voxels only | `calibration_foreground_only.png` / `.json` |
| Low/high-entropy example subjects (paper Figure 1 candidates) | `examples_entropy_extremes/` |
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
