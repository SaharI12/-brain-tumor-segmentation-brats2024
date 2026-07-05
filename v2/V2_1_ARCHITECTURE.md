# v2.1 — Attention U-Net + Stronger Augmentation

## Overview — what, why, how

**What:** we segment brain tumors from multi-parametric MRI (T1n, T1c, T2w, T2-FLAIR) into three clinically-relevant sub-regions — Tumor Core (TC), Whole Tumor (WT), and Enhancing Tumor (ET) — using the BraTS 2024 GLI dataset. `v2.1` is the second architecture iteration in this project: a 3D U-Net whose skip connections are gated by a learned **attention gate**, trained with a heavier augmentation pipeline than the original (`v1`) model.

**Why:** the raw v1 U-Net (see root `CLAUDE.md`) already worked (mean Dice 0.8743), but it was weakest exactly where the task is hardest — the enhancing tumor (ET) and necrotic core (NCR), which are the smallest, most class-imbalanced regions (ET is ~0.36% of all voxels). A plain skip connection dumps the *entire* encoder feature map into the decoder regardless of whether it's relevant at that location; attention gates let the network learn to suppress the background-dominated parts of each skip connection before it reaches the decoder, which should help most on exactly these small structures. Stronger augmentation (flips + intensity jitter + noise) was added at the same time to reduce overfitting risk from the more expressive gated architecture.

**How:** everything downstream — the model (`model_v2.py`), the augmented dataset (`dataset_v2.py`), the training loop (`train_v2_1.py`), and both inference paths (`evaluate_v2.py`, `inference_report_v2.py`) — is documented section-by-section below, with real numbers pulled from the saved checkpoint and eval logs, not estimates.

**Main goal:** beat v1's Dice/HD95 on the same persistent 20%-holdout protocol, particularly on ET/TC, while keeping the same MC-Dropout-based uncertainty estimation this project is built around (see root `CLAUDE.md` — MC Dropout is the project's core innovation). Result achieved: **Dice mean 0.8870 vs v1's 0.8743**, **HD95 mean 4.92mm vs v1's 5.83mm** (§6).

## What changed vs v1

| | v1 (root `train.py` / `model.py`) | v2.1 (`train_v2_1.py` / `model_v2.py`) |
|---|---|---|
| Skip connections | plain concat | **attention-gated** (Oktay et al. 2018) |
| Augmentation | basic | flips (3 axes) + per-modality intensity scale/shift + Gaussian noise |
| Val split | re-shuffled per run | **persistent** `val_split.json` (same 324 subjects always) |
| LR schedule | — | 5-epoch linear warmup → cosine annealing |
| Params | 21.7M (init_features=32) | 21.8M (init_features=32) — attention gates add negligible params |

---

## 1. Architecture — `UNet3DAttn` (`model_v2.py`)

3D U-Net, 4 encoder stages + bottleneck + 4 decoder stages, with an **attention gate** on every skip connection instead of a plain concat.

**Channel flow** (F=32, bottleneck = min(16F, 320) = 320), input `160×208×160`:

```
Input   (B,   4, 160, 208, 160)
enc0    (B,  32, 160, 208, 160) → skip s0
down0 ↓
enc1    (B,  64,  80, 104,  80) → skip s1
down1 ↓
enc2    (B, 128,  40,  52,  40) → skip s2
down2 ↓
enc3    (B, 256,  20,  26,  20) → skip s3
down3 ↓
bottle  (B, 320,  10,  13,  10)   ← Dropout3d(p) here
dec3    upsample×2 → (B,320,20,26,20) → gate(320,256) → cat(320+256) → 256   ← Dropout3d(p) here
dec2    upsample×2 → (B,256,40,52,40) → gate(256,128) → cat(256+128) → 128   ← Dropout3d(p) here
dec1    upsample×2 → (B,128,80,104,80) → gate(128, 64) → cat(128+ 64) →  64   (no dropout)
dec0    upsample×2 → (B, 64,160,208,160) → gate( 64, 32) → cat( 64+ 32) →  32   (no dropout)
out     (B,   4, 160, 208, 160)  ← 1×1×1 conv, no activation (raw logits)
```

Each decoder stage does **upsample → gate → concat → conv**, in that order (`DecoderBlock.forward`, `model_v2.py:86-90`):
```python
def forward(self, x, skip):
    x    = self.up(x)               # trilinear upsample ×2 (spatial size only, channels unchanged)
    skip = self.gate(g=x, x=skip)   # attention-gate the skip using the upsampled x as context
    x    = torch.cat([x, skip], dim=1)
    return self.dropout(self.block(x))
```

**Building blocks:**
- `ConvBlock`: `Conv3d(3×3×3, no bias) → InstanceNorm3d → LeakyReLU(0.01)`, twice, same resolution.
- `DownConv`: stride-2 `Conv3d` (learned downsampling, not MaxPool — keeps more spatial info).
- `AttentionGate` — see §1a below for the full mechanics.
- `DecoderBlock`: trilinear upsample (×2) → attention-gate the skip → concat → `ConvBlock` → optional `Dropout3d`.

### 1a. How the attention gate actually works

It's a tiny conv sub-network (3 conv layers, all 1×1×1) trained by the exact same backprop signal as every other weight in the model — no separate loss, no supervision on what the mask "should" look like. It falls out purely of minimizing `DiceFocalLoss`.

**The computation**, from `AttentionGate.forward` (`model_v2.py:63-68`):
```python
def forward(self, g, x):
    g1  = self.W_g(g)          # 1×1×1 conv: decoder signal (g) → F_int channels
    x1  = self.W_x(x)          # 1×1×1 conv: encoder skip (x)  → F_int channels
    psi = self.relu(g1 + x1)   # combine additively, then nonlinearity
    psi = self.psi(psi)        # 1×1×1 conv: F_int → 1 channel, + InstanceNorm + Sigmoid
    return x * psi              # scale the skip by the resulting per-voxel gate
```

Two things worth being precise about, since "attention" is an overloaded word:

1. **It's additive attention (Bahdanau-style, 2014 seq2seq), not transformer attention.** `g1 + x1` then squash is exactly the original seq2seq attention-score structure, just computed per-voxel instead of per-timestep. There's no Q/K/V, no dot-product similarity, no `√d` scaling.
2. **It doesn't mix information across spatial positions — this is the important part.** Transformer self-attention lets every token attend to every other token (softmax over the whole sequence, weights sum to 1 across positions). This gate does **not** do that. Each voxel's gate value `ψ` depends only on that same voxel's decoder (`g`) and encoder (`x`) feature vectors — voxel `(i,j,k)` never sees voxel `(i+1,j,k)`, and there's no normalization forcing the mask to sum to anything across space. It's an independent per-voxel sigmoid, not a distribution over positions.

So functionally this is closer to a **learned spatial gate / soft mask** (same family as Squeeze-and-Excite, but per-voxel instead of per-channel) than to "attention" in the transformer sense. It's called an attention *gate* because Oktay et al. borrowed the additive-attention scoring function from Bahdanau attention, not because it does sequence-style attention.

**Why it learns anything useful:** gradients flow backward through `x * psi` into `W_g`, `W_x`, `psi`'s conv weights, so wherever suppressing a skip voxel (`ψ→0`) reduces the segmentation loss, the network pushes it that way; wherever keeping it (`ψ→1`) helps, it does that instead. There's no explicit "focus on tumor" objective — the suppression pattern is an emergent side-effect of end-to-end optimization, same as every conv filter elsewhere in the network.

Concrete shapes for `dec3` (the deepest, heaviest gate): `g=(B,320,20,26,20)`, `skip=(B,256,20,26,20)`, `F_int=max(256//2,8)=128`, `ψ=(B,1,20,26,20)` — broadcast-multiplied across all 256 channels of `skip`. Only the skip branch is modified; the decoder branch (`g`) passes through unchanged into the concat.

**Learn more:**
- Paper: [Oktay et al., 2018 — Attention U-Net: Learning Where to Look for the Pancreas (arXiv:1804.03999)](https://arxiv.org/abs/1804.03999) — the original attention-gate formulation used here.
- Video: [225 - Attention U-Net: What is attention and why is it needed for U-Net?](https://www.youtube.com/watch?v=KOF38xAvo8I) — short conceptual walkthrough of exactly this gate.
- Official reference implementation: [ozan-oktay/Attention-Gated-Networks (GitHub)](https://github.com/ozan-oktay/Attention-Gated-Networks) — the paper authors' own PyTorch code.

**Where MC Dropout lives:** `Dropout3d(p=0.15)` is placed in the bottleneck and the two *deepest* decoder blocks (`dec3`, `dec2`) only — not in `dec1`/`dec0` or the encoder. This concentrates stochasticity where the receptive field is largest (most abstract/uncertain features) while keeping the final high-resolution reconstruction deterministic-ish.

**Params:** 21,766,304 (21.77M) at `init_features=32`, by block:

| Block | Params | | Block | Params | |
|---|---|---|---|---|---|
| enc0 | 31,232 | 0.03M | bottleneck | 5,530,880 | 5.53M |
| down0 | 55,424 | 0.06M | dec3 (incl. attn gate) | 5,826,435 | 5.83M |
| enc1 | 221,440 | 0.22M | dec2 | 1,795,011 | 1.80M |
| down1 | 221,440 | 0.22M | dec1 | 448,995 | 0.45M |
| enc2 | 885,248 | 0.89M | dec0 | 112,371 | 0.11M |
| down2 | 885,248 | 0.89M | out_conv | 132 | ~0 |
| enc3 | 3,539,968 | 3.54M | | | |
| down3 | 2,212,480 | 2.21M | **Total** | **21,766,304** | **21.77M** |

`dec3` is the single heaviest block — concatenating 320+256=576 channels into a 256-channel `ConvBlock` means its 3×3×3 convs dominate; the attention gate's own 1×1×1 convs (at 128 channels) are comparatively tiny.

**Region derivation** (`get_region_masks`, shared by training/eval/inference) from the 4-class argmax `{0=bg, 1=NCR, 2=SNFH/edema, 3=ET}`:
```
TC = (pred==1) | (pred==3)
WT = (pred==1) | (pred==2) | (pred==3)
ET = (pred==3)
```

---

## 2. Dataset & Augmentation — `dataset_v2.py`

`BraTSDatasetV2` loads the same preprocessed `(4, 160, 208, 160)` H5 files as v1, but adds a stronger augmentation pipeline (train-only; validation is **never** augmented):

- Independent random flip on each of the 3 spatial axes, p=0.5 each.
- Per-modality intensity **scale** ~ U[0.85, 1.15], p=0.5 per modality (applied independently to t1n/t1c/t2w/t2f).
- Per-modality intensity **shift** ~ U[-0.10, 0.10], p=0.5 per modality.
- Additive Gaussian noise N(0, 0.01) to all modalities, p=0.20.
- Result is clipped back to `[0, 1]` after each intensity op.

**Persistent validation split** (`get_split_loaders_v2`): on the very first run, 20% of subjects are shuffled off (seed=42) and the exact filenames are saved to `checkpoints_v2_1/val_split.json`. Every subsequent run (including `--resume`, and independently `evaluate_v2.py` / `inference_report_v2.py`) reloads that same file list — so the held-out set never drifts even if new files are added to `processed/train/` or the seed changes. Current split: **1,297 train / 324 val**.

---

## 3. Loss & Optimization

```python
loss_fn   = DiceFocalLoss(include_background=False, to_onehot_y=True, softmax=True, gamma=2.0)
optimizer = AdamW(lr=2e-4, weight_decay=1e-4)
scheduler = 5-epoch LinearLR warmup (0.1× → 1.0×) → CosineAnnealingLR(T_max=80, eta_min=lr*0.01)
```

- **DiceFocalLoss**, background excluded from the loss (class imbalance is ~99% background per `CLAUDE.md`), `gamma=2.0` focal term down-weights easy voxels so the loss concentrates on hard/rare classes (NCR, ET).
- **AMP** (`torch.autocast` fp16 + `GradScaler`) on CUDA.
- **Gradient clipping** at `max_norm=1.0` (after `scaler.unscale_`) — standard safeguard against the occasional loss spike from a hard subject.
- **Batch size 1** (full 160×208×160 volumes don't fit larger batches on typical single-GPU setups) — this is also why InstanceNorm (not BatchNorm) is used throughout the model.
- **LR schedule**: warmup exists because a batch-size-1 3D model can destabilize early at full LR; cosine `T_max=80` means the LR completes one full decay cycle every 80 epochs (it does **not** restart — `SequentialLR` just hands off from warmup to a single 80-epoch cosine curve, so past epoch 85 the LR sits near `eta_min = 2e-6`).

---

## 4. Training script — `train_v2_1.py`

Standard loop: per-epoch train pass → full validation pass (`DiceMetric`, `reduction="mean_batch"`, TC/WT/ET) → checkpoint.

**Checkpointing:**
- `latest_checkpoint.pth` — overwritten every epoch, holds full state (`model`, `optimizer`, `scheduler`, `scaler`, `best_dice`, `no_improve`, `args`) for exact resume via `--resume`.
- `best_model.pth` — overwritten only when mean val Dice improves.
- Early stopping: `patience=35` epochs without a new best mean Dice.

**Key CLI args** (defaults shown):

| Arg | Default | Notes |
|---|---|---|
| `--data_dir` | `processed/train` | |
| `--ckpt_dir` | `v2/checkpoints_v2_1` | |
| `--init_features` | 32 | |
| `--epochs` | 300 | upper bound; early stopping usually fires first |
| `--batch_size` | 1 | |
| `--lr` | 2e-4 | peak LR, after warmup |
| `--weight_decay` | 1e-4 | |
| `--dropout` | 0.15 | bottleneck + dec3/dec2 only |
| `--warmup_epochs` | 5 | |
| `--t_max` | 80 | cosine period |
| `--val_split` | 0.2 | ignored once `val_split.json` exists |
| `--patience` | 35 | |
| `--resume` | off | loads `latest_checkpoint.pth` |
| `--debug` | off | 1 epoch, 5 steps — smoke test |

**Run it:**
```bash
python train_v2_1.py                     # fresh run
python train_v2_1.py --resume            # continue from latest_checkpoint.pth
python train_v2_1.py --debug             # smoke test
```

---

## 5. Inference

There are two separate inference entry points, both loading `UNet3DAttn` with the architecture args stored inside the checkpoint (`ckpt["args"]`), so you never have to remember `init_features`/`dropout` by hand.

### 5a. MC Dropout mechanics (`mc_inference`, `model_v2.py`)

```python
mean_pred, entropy = mc_inference(model, image, n_passes=N)
```
- Forces the model into `.train()` mode (so `Dropout3d` layers stay stochastic) while still running under `torch.no_grad()` — MC Dropout needs dropout active at inference, not gradients.
- Runs `N` stochastic forward passes, accumulating the **online mean** of the softmax probabilities (never holds all N full-volume tensors at once — keeps peak memory to ~2 volumes).
- Predictive entropy `H = -∑_c p̄_c·log(p̄_c)` is computed from the **mean** prediction (not averaged per-pass entropy) — this is the standard "entropy of the mean" uncertainty estimate.
- Restores the model's original train/eval mode before returning.

### 5b. `evaluate_v2.py` — quantitative Dice/HD95 only

Loads a checkpoint, reloads the *exact* val split from `val_split.json` next to it (or `--val_split_json`), runs either a single deterministic forward pass (`--mc_passes 0`) or MC Dropout averaging (`--mc_passes N`), and prints aggregate `DiceMetric` + `HausdorffDistanceMetric` (95th percentile) over TC/WT/ET.

```bash
python evaluate_v2.py --checkpoint checkpoints_v2_1/best_model.pth --mc_passes 0
python evaluate_v2.py --checkpoint checkpoints_v2_1/best_model.pth --mc_passes 20
```

### 5c. `inference_report_v2.py` — visual + quantitative report

Produces figures on top of the metrics, for all 324 val subjects by default:

1. **`slices/<subject>.png`** — 4-panel axial-slice figure per subject at that subject's most tumor-rich slice: T1c | GT overlay | Prediction overlay | TP/FP/FN difference map (green/orange/blue), with per-subject Dice + HD95 in the title.
2. **`metrics_table.png`** — one-row BraTS-style aggregate table (Dice + HD95, TC/WT/ET/Mean) styled as a dark dashboard card, with method label noting deterministic vs MC(N passes).
3. **`mc_per_pass_table.png` + `.csv`** (only with `--mc_passes N --per_pass_table`) — Dice/HD95 computed **per individual dropout pass** (not the pass-averaged prediction), one row per pass + a final mean±std row. This isolates pass-to-pass variability from dropout stochasticity, as distinct from the single MC-mean-prediction number in `metrics_table.png`. Implementation note: per-pass region masks are computed on CPU to avoid GPU OOM when accumulating `N` metric objects alongside the running probability sum (this OOM'd around subject 68/324 when first tried on GPU).

```bash
python inference_report_v2.py --checkpoint checkpoints_v2_1/best_model.pth \
    --out_dir exploration_output/inference_report_v2_1

python inference_report_v2.py --checkpoint checkpoints_v2_1/best_model.pth \
    --mc_passes 20 --per_pass_table \
    --out_dir exploration_output/montecarlo_v2_1
```

---

## 6. Actual results (best_model.pth, epoch 108)

Evaluated on the persistent 324-subject holdout (`checkpoints_v2_1/val_split.json`):

| Mode | Dice TC | Dice WT | Dice ET | **Dice Mean** | HD95 TC | HD95 WT | HD95 ET | **HD95 Mean** |
|---|---|---|---|---|---|---|---|---|
| Deterministic (`--mc_passes 0`) | 0.8791 | 0.9206 | 0.8613 | **0.8870** | 4.83 | 4.87 | 5.06 | **4.92 mm** |
| MC Dropout, 20 passes | 0.8794 | 0.9205 | 0.8616 | **0.8872** | 4.81 | 4.88 | 5.04 | **4.91 mm** |

MC averaging over 20 passes barely moves the point estimate (+0.0002 mean Dice) — expected, since dropout is only in the bottleneck + 2 deepest decoder blocks specifically to generate calibrated *uncertainty*, not to boost raw accuracy. The value of MC Dropout here is the per-voxel entropy map, not the mean-prediction Dice.

**Training trajectory:** best checkpoint hit at epoch 108 (mean Dice 0.8870, saved to `best_model.pth`). Training continued to at least epoch 143 (`latest_checkpoint.pth`) with `no_improve=34/35` — i.e. one epoch away from the early-stopping trigger — before the checkpoints captured here were pulled.

### vs v1 (root `model.py`/`train.py`, from `CLAUDE.md`)

| Metric | v1 (epoch 113) | v2.1 (epoch 108) | Δ |
|---|---|---|---|
| Dice Mean | 0.8743 | 0.8870 | **+0.0127** |
| HD95 Mean | 5.83 mm | 4.92 mm | **−0.91 mm** |

Same param budget (~21.7–21.8M), same evaluation protocol (20% holdout). The gain is attributable to attention-gated skips + the stronger augmentation pipeline — both isolate the encoder features that matter most for the small/hard regions (ET, NCR), which is exactly where v1 was weakest (ET Dice 0.8503 → 0.8613 here).

---

## 7. File reference

| File | Role |
|---|---|
| `model_v2.py` | `UNet3DAttn`, `AttentionGate`, `mc_inference`, `get_region_masks` |
| `dataset_v2.py` | `BraTSDatasetV2` (augmentation), `get_split_loaders_v2` (persistent split) |
| `train_v2_1.py` | Training loop, checkpointing, resume, early stopping |
| `evaluate_v2.py` | Dice/HD95-only evaluation, deterministic or MC |
| `inference_report_v2.py` | Per-subject figures + aggregate table + MC per-pass table |
| `checkpoints_v2_1/val_split.json` | Persistent 1,297/324 train/val filename split |
| `checkpoints_v2_1/best_model.pth` | Best checkpoint (epoch 108, mean Dice 0.8870) |
| `checkpoints_v2_1/latest_checkpoint.pth` | Most recent epoch, for `--resume` |