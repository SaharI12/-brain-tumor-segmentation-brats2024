# BraTS 2024 GLI — Training, Inference & Results Analysis

**Authors:** Sahar Ifrah, Hadas Avraham  
**Task:** Multi-region segmentation of adult diffuse gliomas from multi-parametric MRI  
**Dataset:** BraTS 2024 GLI (post-treatment adult diffuse glioma)

---

## 1. Dataset

| Split | Source | Subjects | Labels |
|-------|--------|----------|--------|
| Training | BraTS-GLI-TrainingData + AdditionalTrainingData | 1,621 | ✓ |
| Validation (official) | BraTS-GLI-ValidationData | 188 | ✗ (hidden) |

**Modalities per subject:** T1 native (T1n), T1 post-contrast (T1c), T2-weighted (T2w), T2-FLAIR (T2f) — all pre-registered to SRI24 atlas and skull-stripped, 1 mm isotropic.

**Label convention (BraTS 2024 GLI):**

| Label | Region |
|-------|--------|
| 0 | Background |
| 1 | Necrotic Tumor Core (NCR) |
| 2 | Surrounding Non-Enhancing FLAIR Hyperintensity / Edema (SNFH) |
| 3 | Enhancing Tumor (ET) — remapped from label 4 where present |

**Evaluation sub-regions:**
- **TC** (Tumor Core) = labels 1 + 3
- **WT** (Whole Tumor) = labels 1 + 2 + 3
- **ET** (Enhancing Tumor) = label 3

**Class imbalance (verified, 30 subjects):**

| Label | % of voxels |
|-------|------------|
| 0 Background | 98.82% |
| 1 NCR | 0.011% |
| 2 SNFH/Edema | 0.816% |
| 3 ET | 0.356% |

**Preprocessing applied:**
1. Normalize each modality by its global max across the training set → values in [0, 1]
2. Remap label 4 → 3 (standardize ET label across the mixed convention in raw data)
3. Tight-crop to brain bounding box (remove all-zero borders using T1n > 0 mask)
4. Center-crop / pad to fixed size **160 × 208 × 160** (both dims divisible by 16 for U-Net pooling)
5. Save as H5 files: `images` float32 (4, 160, 208, 160), `seg` uint8 (160, 208, 160)

---

## 2. Model Architecture

**3D U-Net with Monte Carlo Dropout** (`model.py`)

- **Encoder:** 4 stages, each with a ConvBlock (Conv3d → InstanceNorm3d → LeakyReLU(0.01), ×2) followed by a learnable stride-2 DownConv. Feature channels: 32 → 64 → 128 → 256.
- **Bottleneck:** Conv × 2 + Dropout3d(p=0.2). Spatial resolution: 10×13×10, channels: 320 (capped from 512).
- **Decoder:** 4 stages, each with trilinear upsampling (no checkerboard artifacts) → concatenation with skip connection → ConvBlock. Dropout3d(p=0.2) in the two deepest decoder blocks (dec3, dec2).
- **Output:** Conv3d 1×1×1 → 4 class logits (160×208×160).
- **Parameters:** 21.7M (init_features=32)

**Why InstanceNorm over BatchNorm:** Medical image batch sizes are typically 1 (memory constraint). BatchNorm statistics are unreliable at batch size 1; InstanceNorm normalizes per-sample per-channel, which is robust regardless of batch size.

**Why stride-2 DownConv over MaxPool:** Learnable downsampling retains spatial information through learned weights rather than discarding the maximum activation.

**Why trilinear upsampling over transposed convolution:** Avoids checkerboard artifacts that arise from transposed convolution stride patterns.

---

## 3. Training

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| LR schedule | Cosine annealing (T_max = 300 epochs) |
| Loss function | DiceFocalLoss (γ=2.0, background excluded) |
| Batch size | 1 |
| Dropout probability | 0.2 |
| Early stopping patience | 20 epochs |
| Max epochs | 300 |
| Data split | 80% train / 20% val (seed=42, fixed) |
| Augmentation | Random 90° rotations in the H–D plane (H=D=160) |

**Why DiceFocalLoss:** Combines Dice loss (handles class imbalance globally) with Focal loss (down-weights easy background voxels, forces the model to focus on hard foreground cases). Given ~99% background voxels, pure cross-entropy would trivially predict all-background.

**Training infrastructure:** NVIDIA A100 GPU on RunAI cluster. Mixed precision (AMP, fp16) enabled. Checkpoints saved every epoch (`latest_checkpoint.pth`) for resume after preemption; best checkpoint saved separately (`best_model.pth`).

**Training outcome:** Early stopping triggered at **epoch 113 / 300** (no improvement in mean validation Dice for 20 consecutive epochs). Best checkpoint selected at the epoch with peak validation Dice.

---

## 4. Evaluation Protocol

**Internal validation split:** 20% holdout of the labeled training set (n=324 subjects, same seed=42 as training). These subjects were **never used for gradient updates** but were used for early stopping — meaning the final checkpoint was selected based on their Dice. This is the standard evaluation setup when the official BraTS challenge leaderboard is unavailable.

> Note: The BraTS 2024 challenge closed before leaderboard submission was possible. The 188 official validation subjects have no publicly released ground-truth labels, so the internal holdout is the only available quantitative evaluation.

**Metrics:**
- **DSC** (Dice Similarity Coefficient): measures volumetric overlap. Range [0, 1], 1 = perfect.
- **HD95** (95th-percentile Hausdorff Distance, mm): measures worst-case surface distance after excluding the top 5% outlier boundary points. Lower is better.

Both metrics computed per region (TC, WT, ET) and averaged.

---

## 5. Inference Modes

### 5a. Deterministic Inference
Model set to `eval()` mode — dropout is **disabled**. Single forward pass. Standard inference used in most segmentation pipelines.

### 5b. Monte Carlo Dropout Inference (20 passes)
Model forced into `train()` mode — dropout remains **active**. 20 stochastic forward passes are run; each pass randomly drops different neurons in the bottleneck and top decoder blocks. The 20 softmax probability maps are averaged to produce the final prediction.

**Why this works as uncertainty estimation:** With dropout active, each forward pass samples a different sub-network, approximating a draw from the posterior distribution over model weights (Gal & Ghahramani, 2016). The variance across passes reflects the model's epistemic uncertainty — regions where the model is systematically unsure.

**Uncertainty map:** Per-voxel predictive entropy computed as:

```
H[y|x] = -∑_c p̄_c · log(p̄_c)
```

where p̄_c is the mean softmax probability for class c across all 20 passes. High entropy = high uncertainty.

---

## 6. Results

### 6a. Quantitative Results (Internal Validation, n=324)

| Inference Mode | DSC TC ↑ | DSC WT ↑ | DSC ET ↑ | Mean DSC ↑ | HD95 TC ↓ | HD95 WT ↓ | HD95 ET ↓ | Mean HD95 ↓ |
|----------------|----------|----------|----------|------------|-----------|-----------|-----------|------------|
| Deterministic  | 0.8638   | 0.9087   | 0.8503   | **0.8743** | 5.65      | 6.11      | 5.74      | **5.83**   |
| MC Dropout (20 passes) | 0.8638 | 0.9087 | 0.8503 | **0.8743** | 5.65 | 6.11 | 5.74 | **5.83** |

*(Per-subject mean Dice: 0.8715 deterministic vs 0.8717 MC — difference < 0.0002, negligible)*

**Key observations:**
- **WT** achieves the highest Dice (0.9087) — Whole Tumor is the largest and most consistent region, easiest to delineate.
- **ET** is slightly lower (0.8503) — Enhancing Tumor is a small, heterogeneous region; post-treatment cases make it harder to distinguish ET from treatment-induced enhancement.
- **TC** (0.8638) — Tumor Core (NCR+ET) is affected by the difficulty of both sub-regions.
- MC Dropout averaging over 20 passes produces essentially identical metrics to deterministic inference. **This is expected** — averaging stochastic predictions reduces variance but does not change the mean prediction meaningfully for well-trained models.

### 6b. Value of MC Dropout

The benefit of MC Dropout is **not** improved average accuracy — it is **uncertainty quantification**. Entropy analysis on a held-out subset showed:

| Region | Mean Entropy |
|--------|-------------|
| True Positives (TP) | 0.039 |
| False Positives (FP) | 0.236 |
| False Negatives (FN) | ~0.20 |

The model assigns **6× higher entropy to its own mistakes** (FP regions) compared to correct predictions (TP). This means the uncertainty map is a reliable signal for flagging unreliable predictions — clinically valuable for identifying cases that warrant human review.

---

## 7. Comparison with Published Methods

> **Important caveats before reading this table:**
> 1. **Post-treatment vs pre-treatment:** BraTS 2024 GLI is a post-treatment dataset. All BraTS 2021/2022/2023 results below are on pre-treatment data, which is an entirely different clinical scenario (no surgery cavities, radiation effects, or pseudoprogression). Numbers are **not directly comparable**.
> 2. **Metric definition:** BraTS 2023 and 2024 introduced **lesion-wise** DSC/HD95 (evaluated per individual lesion, then averaged). We compute the classical **scan-level** DSC/HD95 (whole volume). Lesion-wise metrics are generally harder and produce lower numbers.
> 3. **Evaluation set:** Our results are on an **internal holdout** (20% of training set, n=324). All challenge results below are from the **official leaderboard** evaluated on hidden test/validation sets.

### 7a. BraTS 2024 GLI — Post-Treatment (Same Task)

| Method | DSC TC ↑ | DSC WT ↑ | DSC ET ↑ | Eval set | Metric |
|--------|----------|----------|----------|----------|--------|
| **Ours — 3D U-Net + MC Dropout** | **0.8638** | **0.9087** | **0.8503** | Internal holdout (n=324) | Scan-level |
| BraTS 2024 challenge submission † | 0.7499 | 0.9055 | 0.8124 | Official validation | Lesion-wise |

† Representative result from the BraTS 2024 challenge paper (Court et al., arXiv 2405.18368). Lesion-wise DSC is structurally different from scan-level DSC — lower values do not imply a worse model.

### 7b. Context — Pre-Treatment Glioma (Different Task, Not Directly Comparable)

Shown for orientation only. These methods were trained and evaluated on pre-treatment glioma data (BraTS 2021/2023) where tumors appear substantially different.

| Method | DSC TC ↑ | DSC WT ↑ | DSC ET ↑ | HD95 TC ↓ | HD95 WT ↓ | HD95 ET ↓ | Dataset |
|--------|----------|----------|----------|-----------|-----------|-----------|---------|
| BraTS 2023 winner (ensemble + augmentation) † | 0.8673 | 0.9005 | 0.8509 | 14.47 | — | 17.70 | BraTS 2023 val (lesion-wise) |
| nnUNetFormer (Gao et al.) | 0.921 | 0.936 | 0.872 | 4.57 | 3.96 | 10.45 | BraTS 2021 (scan-level) |
| SwinUNETR (Hatamizadeh et al.) | 0.831 | 0.852 | 0.799 | — | — | — | BraTS 2021 (scan-level) |
| **Ours — 3D U-Net + MC Dropout** | **0.8638** | **0.9087** | **0.8503** | **5.65** | **6.11** | **5.74** | BraTS 2024 internal (scan-level) |

† BraTS 2023 winner: "How we won BraTS 2023 Adult Glioma challenge? Just faking it!" (arXiv 2402.17317) — used synthetic data augmentation and model ensembling.

**Takeaway:** Our single-model 3D U-Net sits between SwinUNETR and nnUNetFormer on pre-treatment benchmarks in terms of raw numbers, while operating on the harder post-treatment task. The HD95 values (5.65–6.11 mm) are competitive with scan-level methods on BraTS 2021.

---

## 8. Discussion

**Strengths:**
- Strong WT Dice (0.9087) — reliable delineation of the full tumor extent.
- MC Dropout provides calibrated uncertainty at no inference-time training cost (dropout layers already exist for regularization).
- Lightweight relative to ensemble methods — 20 passes of one model vs. training and storing N separate models.

**Limitations:**
- Evaluation is on an internal holdout only. Leaderboard comparison against other BraTS 2024 submissions is not possible.
- Post-treatment gliomas are inherently harder to segment than pre-treatment (treatment effects, pseudoprogression) — direct comparison with BraTS 2021/2023 results is not valid.
- The 20% holdout was used for early stopping (model selection), so it is not a fully independent test set.

**Future directions:**
- Integrate uncertainty maps into the clinical workflow — flag high-entropy predictions for radiologist review.
- Explore test-time augmentation (TTA) as a complementary uncertainty estimation method.
- Train on BraTS 2023 pre-treatment data to assess cross-dataset generalization.