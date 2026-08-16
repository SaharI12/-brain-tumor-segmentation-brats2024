# Project Progress

## Goal
Train a 3D U-Net with Monte Carlo Dropout to segment brain tumors (NCR, SNFH/Edema, ET) from multi-parametric MRI using the BraTS 2024 Adult Glioma (post-treatment) dataset.

---

## ✅ All Phases Complete

### 1. Data Exploration (`data_playing.py`)
- Confirmed raw volume shape: **182 × 218 × 182**, 1 mm isotropic, float32
- Identified **mixed ET labeling convention**: labels 3 and 4 both mean ET (75% of subjects have both). Resolved by remapping 4 → 3.
- Verified no NaN/Inf values, no shape inconsistencies across subjects
- Confirmed severe class imbalance: background = 98.82%, NCR = 0.011%, SNFH = 0.816%, ET = 0.356%

### 2. Preprocessing (`preprocess.py`)
- **Normalization:** each modality divided by its global max across the training set → [0, 1]
- **Label remap:** label 4 → 3 (unified ET class), final labels: {0, 1, 2, 3}
- **Tight crop:** removed all-zero border slices per subject using T1n brain mask
- **Fixed-size output:** center-crop/pad to **160 × 208 × 160** (divisible by 16 for U-Net pooling)
- **Output:** 1,621 train H5 files + 188 val H5 files
- Normalization stats: `processed/normalization_stats.json`

### 3. PyTorch Dataset + DataLoader (`dataset.py`)
- `BraTSDataset` reads H5 files, returns `images (4, 160, 208, 160)` and `seg (160, 208, 160)`
- 80/20 train/val split by subject (seed=42, fixed) via `get_split_loaders()` → **1,297 train / 324 val**
- Augmentation: random 90° rotations in the H–D plane (H=D=160, shape-preserving)

### 4. 3D U-Net Model (`model.py`)
- 4 encoder stages + bottleneck + 4 decoder stages, skip connections via concatenation
- **Dropout3d(p=0.2)** in bottleneck and top-2 decoder blocks (dec3, dec2)
- Learnable stride-2 downsampling (DownConv), trilinear upsampling (no checkerboard artifacts)
- InstanceNorm3d throughout (robust at batch size 1)
- **21.7 M parameters** (init_features=32)
- `mc_inference()` function for Monte Carlo Dropout inference

### 5. Training (`train.py`)
- **Loss:** DiceFocalLoss (γ=2.0, background excluded) — handles severe class imbalance
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-5)
- **Schedule:** Cosine annealing (T_max=300)
- **Hardware:** NVIDIA A100 on RunAI cluster, mixed precision (AMP fp16)
- **Early stopping:** patience=20 epochs
- **Outcome:** stopped at **epoch 113 / 300**, best mean val Dice **0.8743**
- Checkpoints: `best_model.pth` + `latest_checkpoint.pth` (resume support)

### 6. Evaluation (`evaluate.py`)
- Dice + HD95 per BraTS sub-region (TC, WT, ET)
- Supports `--mc_passes N` for MC Dropout inference
- **Final results (internal validation, n=324):**

| Metric | TC | WT | ET | Mean |
|--------|----|----|----|----|
| Dice   | 0.8638 | 0.9087 | 0.8503 | **0.8743** |
| HD95 (mm) | 5.65 | 6.11 | 5.74 | **5.83** |

### 7. Monte Carlo Dropout — Uncertainty Analysis (`visualize_uncertainty.py`)
- 20 stochastic forward passes with dropout active (`model.train()` at inference)
- Per-voxel predictive entropy: H[y|x] = −∑ p̄_c log(p̄_c)
- **Key finding:** FP entropy (0.236) >> TP entropy (0.039) — model correctly flags its own mistakes with 6× higher uncertainty
- MC Dropout Dice (0.8717) ≈ Deterministic Dice (0.8715) — value is uncertainty estimation, not accuracy gain

### 8. Inference Report (`inference_report.py`)
- Per-subject figure: best tumor-containing axial slice, 4 panels (T1c | GT | Prediction | Diff map)
- Aggregate metrics table: BraTS-style DSC + HD95, labeled as internal validation
- Supports `--mc_passes N` to switch to MC Dropout inference
- **Outputs:**
  - `exploration_output/inference_report/` — deterministic results (324 figures + table)
  - `exploration_output/montecarlo/` — MC Dropout results (324 figures + table)
  - `results/` — key figures committed to git

### 9. Submission Predictions (`predict_submission.py`)
- Inference on 188 official BraTS 2024 validation subjects (no GT labels)
- Predictions restored to original 182×218×182 space with correct affine/header
- Saved as `predictions/<subject>-seg.nii.gz` → `predictions.zip`
- Note: BraTS 2024 challenge closed before leaderboard submission was possible

### 10. Architecture Diagram (`draw_architecture.py`)
- Paper-style 3D U-Net diagram: encoder / bottleneck / decoder blocks, skip connections, MC Dropout highlights
- Saved to `results/architecture.png`

---

## Final Results Summary

| | Dice TC | Dice WT | Dice ET | Mean Dice | HD95 TC | HD95 WT | HD95 ET | Mean HD95 |
|---|---|---|---|---|---|---|---|---|
| Deterministic | 0.8638 | 0.9087 | 0.8503 | 0.8743 | 5.65 | 6.11 | 5.74 | 5.83 |
| MC Dropout (20 passes) | 0.8638 | 0.9087 | 0.8503 | 0.8743 | 5.65 | 6.11 | 5.74 | 5.83 |

Evaluated on internal 20% holdout of BraTS 2024 GLI training set (n=324).  
See `ANALYSIS.md` for full discussion and comparison with published methods.

---

## ✅ v2.1 — Attention U-Net (`v2/`)

Second-generation model: same 4-modality input / MC Dropout innovation, upgraded with
attention-gated skip connections, a stronger augmentation pipeline, and a persistent
validation split. See `v2/V2_1_ARCHITECTURE.md` for the full architecture writeup.

### 11. Attention U-Net Model & Training (`model_v2.py`, `dataset_v2.py`, `train_v2_1.py`)
- **Architecture:** `UNet3DAttn` — identical encoder/decoder topology to `model.py`'s `UNet3D`, but each
  skip connection passes through a soft attention gate (Oktay et al. 2018) gated by the decoder signal —
  21.8M params (init_features=32)
- **Augmentation:** random flips on all 3 axes, per-modality intensity scale/shift, Gaussian noise (`dataset_v2.py`)
- **Persistent val split:** saved to `checkpoints_v2_1/val_split.json` on first run, reloaded on every
  resume — guarantees the same 324 val subjects across the whole training run (unlike v1's reshuffle-by-seed)
- **Training:** AdamW (lr=2e-4), 5-epoch linear warmup → cosine annealing (T_max=80), patience=35
- **Outcome:** best checkpoint at **epoch 108** (mean val Dice 0.8870); training was interrupted by a
  machine reboot at **epoch 143/300** (34/35 epochs without improvement — one epoch short of early
  stopping), so `latest_checkpoint.pth` is resumable but not yet fully converged

### 12. Evaluation (`evaluate_v2.py`)
- Same Dice + HD95 methodology as `evaluate.py`, adapted for `UNet3DAttn` and the persistent val split
- **Deterministic vs MC Dropout (20 passes), internal validation n=324:**

| | Dice TC | Dice WT | Dice ET | Mean Dice | HD95 TC | HD95 WT | HD95 ET | Mean HD95 |
|---|---|---|---|---|---|---|---|---|
| Deterministic | 0.8791 | 0.9206 | 0.8613 | **0.8870** | 4.83 | 4.87 | 5.06 | **4.92** |
| MC Dropout (20-pass mean) | 0.8794 | 0.9205 | 0.8616 | **0.8872** | 4.81 | 4.88 | 5.04 | **4.91** |

- MC Dropout and deterministic inference are essentially identical here (Dice within 0.0003, HD95 within
  0.02mm) — dropout at inference isn't hurting accuracy, so entropy-based uncertainty maps reflect genuine
  predictive uncertainty rather than a degraded mean prediction

### 13. Inference Report + Per-Pass Stability (`inference_report_v2.py`)
- Per-subject figure: best tumor-containing axial slice, 4 panels (T1c | GT | Prediction | Diff map)
- Aggregate BraTS-style metrics table (same style as v1's `inference_report.py`)
- New for v2.1 — `--mc_passes N --per_pass_table`: tracks Dice/HD95 **per individual dropout pass**
  (not just the pass-averaged prediction), aggregated over all 324 val subjects, to quantify pass-to-pass
  stochastic variability:

| MC Dropout, 20 individual passes (n=324 each) | Dice TC | Dice WT | Dice ET | Dice Mean | HD95 TC | HD95 WT | HD95 ET | HD95 Mean |
|---|---|---|---|---|---|---|---|---|
| **Mean ± Std across passes** | 0.8783±0.0005 | 0.9200±0.0002 | 0.8605±0.0005 | 0.8863±0.0004 | 4.89±0.13 | 5.01±0.11 | 5.11±0.13 | 5.00±0.12 |

  Full 20-row per-pass table: `v2/montecarlo_v2_1/mc_per_pass_table.png` / `.csv`. Pass-to-pass std is tiny
  relative to the mean — the model is stable under dropout stochasticity, so each individual pass is
  already about as accurate as the mean prediction.
- **Outputs** (moved out of the gitignored `exploration_output/` so key results are versioned):
  - `v2/inference_report_v2_1/` — deterministic results (324 figures + `metrics_table.png`)
  - `v2/montecarlo_v2_1/` — MC Dropout results (324 figures + `metrics_table.png` + `mc_per_pass_table.png`/`.csv`)

### v2.1 vs v1 — Head to Head (accuracy + uncertainty)

| Metric | v1 — Plain 3D U-Net | v2.1 — Attention U-Net |
|---|---|---|
| Dice TC | 0.8638 | **0.8791** |
| Dice WT | 0.9087 | **0.9206** |
| Dice ET | 0.8503 | **0.8613** |
| Dice Mean | 0.8743 | **0.8870** |
| HD95 TC (mm) | 5.65 | **4.83** |
| HD95 WT (mm) | 6.11 | **4.87** |
| HD95 ET (mm) | 5.74 | **5.06** |
| HD95 Mean (mm) | 5.83 | **4.92** |
| Entropy TN | not computed | 0.0002 |
| Entropy TP | 0.039 | 0.0285 |
| Entropy FP | 0.236 | 0.1960 |
| Entropy FN | ~0.20 | 0.1112 |
| FP/TP entropy ratio | ~6.0× | ~6.9× |
| ECE (calibration) | not computed | 0.0080 |
| AUROC (entropy as error detector) | not computed | 0.8956 |
| Eval set | own 80/20 holdout, reshuffled split | persistent 324-subject val split |

v2.1 improves on every accuracy metric (+1.3 pts mean Dice, ~16% lower mean HD95) and preserves the
qualitative uncertainty behavior (FP entropy » TP entropy, ratio even slightly higher: 6.0× → 6.9×) —
this is from a best checkpoint at epoch 108 of an interrupted run, not yet a fully early-stopped,
converged model the way v1's 113-epoch result is. **Caveats on the comparison:** v1's entropy numbers
are quoted from `ANALYSIS.md` §6b (script output not regenerated on disk — gitignored) and were
measured on v1's own reshuffled holdout, not the persistent split v2.1 uses; v1 has no TN entropy, ECE,
or AUROC figures because `visualize_uncertainty.py` didn't compute a calibration analysis — only
`visualize_uncertainty_v2.py` added ECE/AUROC. So the Dice/HD95 rows are a clean head-to-head, but the
entropy/ECE/AUROC rows are directionally comparable, not a controlled rerun on identical data.

### 14. Uncertainty Maps & Calibration (`visualize_uncertainty_v2.py`)
- Ported v1's `visualize_uncertainty.py` to v2.1 — `UNet3DAttn`, `checkpoints_v2_1/best_model.pth`,
  and the persistent `val_split.json` (instead of v1's reshuffle-by-seed sampling)
- **Examples** (10 subjects, 20 MC passes, one paper-style figure per subject, black background):
  same 5-panel layout as v1 — T1c | T1c+GT | T1c+MC-Pred | Entropy map | T1c+Entropy overlay
- **Entropy by whole-tumor region** (TN/TP/FP/FN, aggregated over the 10 example subjects):

| Region | Mean Entropy |
|--------|-------------|
| TN | 0.0002 |
| TP | 0.0285 |
| FP | 0.1960 |
| FN | 0.1112 |

  FP entropy is **~6.9× higher** than TP entropy — consistent with v1's ~6× finding; the model still
  flags its own mistakes reliably after the attention-gate upgrade.
- **New — voxel-level calibration analysis** (50 val subjects, subsampled, restricted to brain-mask
  voxels to avoid trivial background dominating the stats):
  - **Expected Calibration Error (ECE) = 0.0080** — softmax confidence is well calibrated against
    empirical accuracy
  - **AUROC = 0.8956** — using per-voxel entropy alone to predict "this voxel is misclassified"
    substantially beats chance, i.e. entropy is a strong, usable signal for flagging unreliable
    predictions for review
- **Outputs**: `v2/uncertainty_vis_v2_1/{examples/example_NN_<subject>.png (x10), uncertainty_stats.png,
  calibration.png, summary.json}`

### 15. Baseline Comparison vs. Published Model (`run_baseline_segresnet.py`, `compare_baseline.py`)
- Downloaded MONAI's published `brats_mri_segmentation` bundle (SegResNet, Myronenko 2018, trained on
  BraTS 2018) — the only publicly downloadable pretrained BraTS checkpoint found with genuine trained
  dropout (`dropout_prob=0.2`); SwinUNETR/nnU-Net BraTS21 alternatives have no usable dropout
- Built an adapter pipeline (channel reorder, scale-invariant re-normalization, sliding-window
  inference, per-channel binary-entropy MC Dropout) and evaluated it on our **same 324-subject
  persistent val split** used for v2.1 — see `v2/BASELINE_COMPARISON.md` for full methodology

| Model — Mode | Dice Mean | HD95 Mean |
|---|---|---|
| v2.1 (ours) — Deterministic | **0.8870** | **4.92** |
| v2.1 (ours) — MC Dropout (20) | 0.8869 | 4.92 |
| SegResNet (published, BraTS 2018) — Deterministic | 0.5791 | 18.09 |
| SegResNet (published, BraTS 2018) — MC Dropout (20) | 0.5860 | 17.13 |

- The published baseline substantially underperforms on our data (mean Dice 0.58 vs 0.887) — expected,
  since it has never seen BraTS 2024's post-treatment scans (surgical cavities, radiation effects,
  pseudoprogression absent from its BraTS 2018 pre-treatment training data). ET is hit hardest (0.396
  vs 0.861), WT least (0.780 vs 0.921)
- **Uncertainty comparison**: both models show the expected TN < TP < FN ≤ FP entropy ordering, but
  ours discriminates errors far more sharply — FP entropy is **~6.9×** TP entropy for us vs only
  **~2.6×** for the baseline (whole-tumor/WT-channel comparison) — i.e. dropout-based uncertainty
  transfers qualitatively across domains but is much blunter on a model never trained on this data
- **Outputs**: `v2/baseline_comparison/{comparison_table.png, entropy_comparison.png,
  segresnet_results.json}`. Full caveats (domain shift, structurally different entropy formulas,
  single vs. multiple dropout sites) in `v2/BASELINE_COMPARISON.md`

---

## Key Files

| File | Purpose |
|------|---------|
| `data_playing.py` | Raw data exploration (keep as reference) |
| `preprocess.py` | Preprocessing pipeline → H5 files |
| `dataset.py` | PyTorch Dataset + DataLoader |
| `model.py` | 3D U-Net + MC Dropout + `mc_inference()` |
| `train.py` | Training loop |
| `evaluate.py` | Fast console-only evaluation |
| `inference_report.py` | Per-subject figures + aggregate metrics table |
| `visualize_uncertainty.py` | MC Dropout entropy maps |
| `predict_submission.py` | NIfTI predictions for BraTS val subjects |
| `draw_architecture.py` | Architecture diagram |
| `ANALYSIS.md` | Full results analysis and comparison with literature |
| `v2/model_v2.py` | Attention U-Net (`UNet3DAttn`) + `mc_inference()` |
| `v2/dataset_v2.py` | Dataset with stronger augmentation + persistent val split |
| `v2/train_v2_1.py` | v2.1 training loop (tuned hyperparameters) |
| `v2/evaluate_v2.py` | Fast console-only evaluation for v2.1 |
| `v2/inference_report_v2.py` | Per-subject figures + aggregate table + per-pass MC stability table |
| `v2/visualize_uncertainty_v2.py` | v2.1 MC Dropout entropy maps + voxel-level calibration (ECE, AUROC) |
| `v2/run_baseline_segresnet.py` | Runs MONAI's published SegResNet BraTS bundle (+ MC Dropout) on our val split |
| `v2/compare_baseline.py` | Assembles the v1 / v2.1 / SegResNet Dice-HD95 and entropy comparison figures |
| `v2/V2_1_ARCHITECTURE.md` | v2.1 architecture writeup (attention gate mechanics, training setup) |
| `v2/BASELINE_COMPARISON.md` | Methodology + results for the published-model MC comparison |