# Project Progress

## Goal
Train a 3D U-Net with Monte Carlo Dropout to segment brain tumors (NCR, SNFH/Edema, ET) from multi-parametric MRI using the BraTS 2024 Adult Glioma dataset.

---

## ✅ Done

### 1. Data Exploration (`data_playing.py`)
- Confirmed raw volume shape: **182 × 218 × 182**, 1 mm isotropic, float32
- Identified **mixed ET labeling convention**: labels 3 and 4 both mean ET (75% of subjects have both). Resolved by remapping 4 → 3.
- Verified no NaN/Inf values, no shape inconsistencies across subjects
- Confirmed severe class imbalance: background = 98.82%, NCR = 0.011%, SNFH = 0.816%, ET = 0.356%
- Saved visualizations: `exploration_output/`

### 2. Preprocessing (`preprocess.py`)
- **Normalization:** divided each modality by its global max across the training set → [0, 1]
- **Label remap:** label 4 → 3 (unified ET class), final labels: {0, 1, 2, 3}
- **Tight crop:** removed all-zero border slices per subject using t1n brain mask
- **Fixed-size output:** center-crop/pad to **160 × 208 × 160** (divisible by 16, fits all brains without tissue loss)
- **Output:** 1,621 train H5 files + 188 val H5 files, each ~17–22 MB compressed
- Normalization stats saved to `processed/normalization_stats.json`
- Verified on 30 random files: correct shapes, ranges [0,1], labels {0,1,2,3}, no anomalies

### 3. Visualization (`visualize_processed.py`)
- 10 random training examples visualized
- Layout: 3 rows (Axial, Coronal, Sagittal) × 6 columns (T1n, T1c, T2w, T2f, Seg, T1c+Seg overlay)
- Slices centered on tumor centroid so mask is always visible
- Saved to: `exploration_output/processed_vis/`

---

## ⬜ Still To Do

### 4. PyTorch Dataset + DataLoader
- Write a `BraTSDataset` class that reads from the processed H5 files
- Implement random 80/20 train/validation split (by subject, done at training time)
- Apply rotation augmentation (decided: rotation only, no flips or intensity jitter)
- Output per batch: `images (B, 4, 160, 208, 160)`, `seg (B, 160, 208, 160)`

### 5. 3D U-Net Model
- Standard encoder–decoder architecture with skip connections
- Add **dropout layers** in both encoder and decoder blocks (for MC Dropout)
- Input: 4 channels (t1n, t1c, t2w, t2f)
- Output: 4-class segmentation logits `(B, 4, 160, 208, 160)`
- Decide: dropout probability and placement (after conv blocks vs. at bottleneck only)

### 6. Training Loop
- **Loss:** Dice Focal Loss (handles class imbalance)
- **Optimizer:** AdamW
- **Schedule:** cosine annealing
- **Hardware:** cloud GPU (H5 files to be uploaded)
- Track: train loss, val Dice per class (NCR, SNFH, ET) per epoch
- Save best model checkpoint by val Dice

### 7. Evaluation
- Compute **Dice score** and **95th Percentile Hausdorff Distance (HD95)** per region:
  - TC (Tumor Core) = NCR + ET = labels {1, 3}
  - WT (Whole Tumor) = all tumor = labels {1, 2, 3}
  - ET (Enhancing Tumor) = label {3}
- Compare against BraTS 2024 leaderboard baselines

### 8. Monte Carlo Dropout — Uncertainty Analysis
- At inference: run N=10–20 forward passes with dropout active (`model.train()`)
- Compute per-voxel **predictive entropy** across passes → uncertainty map
- Validate: high-uncertainty voxels should spatially overlap tumor boundaries and heterogeneous ET regions
- Visualize uncertainty maps alongside segmentation predictions

---

## Open Decisions
| Topic | Status |
|-------|--------|
| Dropout probability | Not decided yet |
| Dropout placement (every block vs. bottleneck only) | Not decided yet |
| Number of MC passes at inference (N) | Tentatively 10–20 |
| Batch size (depends on GPU VRAM) | Not decided yet |
| Number of training epochs | Not decided yet |
| Cloud GPU provider | Not decided yet |
