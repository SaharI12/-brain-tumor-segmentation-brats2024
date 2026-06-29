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