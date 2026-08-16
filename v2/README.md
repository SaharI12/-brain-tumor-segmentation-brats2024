# v2 — Attention U-Net (BraTS 2024 GLI)

This folder is the second generation of the project's segmentation model. If you're picking this
up for the first time, read this file top to bottom — it's the map. The deep-dive docs
(architecture mechanics) are linked inline rather than repeated here.

**Start here if you only have five minutes:** §1 (what changed and why) and §2 (results table).

---

## 0. How this relates to the root project

The root of the repo (`preprocess.py`, `model.py`, `train.py`, ...) is **v1** —
a plain 3D U-Net, fully documented in the root `CLAUDE.md` and `PROGRESS.md`. It hit mean Dice
0.8743 and established the project's core innovation: **Monte Carlo Dropout** for uncertainty
estimation (dropout stays active at inference; entropy across stochastic passes flags likely
mistakes).

`v2/` reuses the same preprocessed data (`processed/train/*.h5`, `(4,160,208,160)` per subject —
see root `CLAUDE.md` for the full preprocessing pipeline and label convention) and the same MC
Dropout idea, but swaps in a better architecture and a stronger augmentation pipeline. Everything
in this folder is self-contained — its own model, dataset, training loop, and checkpoints, separate
from the root's.

---

## 1. What changed vs v1, and why

v1's plain U-Net was already solid, but weakest exactly where the task is hardest: enhancing tumor
(ET) and necrotic core (NCR), the smallest and most class-imbalanced regions (ET is ~0.36% of all
voxels — see root `CLAUDE.md`). A plain skip connection dumps the *entire* encoder feature map into
the decoder regardless of relevance; **attention gates** (Oktay et al. 2018) let the network learn
to suppress the background-dominated parts of each skip connection before it reaches the decoder.

| | v1 | v2.1 (this folder, current best) |
|---|---|---|
| Skip connections | plain concat | **attention-gated** |
| Augmentation | basic | flips (3 axes) + per-modality intensity scale/shift + Gaussian noise |
| Val split | reshuffled per run | **persistent** (`checkpoints_v2_1/val_split.json`, same 324 subjects always) |
| LR schedule | — | 5-epoch linear warmup → cosine annealing |
| Params | 21.7M | 21.8M (attention gates add negligible params) |

Full mechanics (including a precise note on why this is *not* transformer-style attention — no
Q/K/V, no cross-position mixing, it's a per-voxel learned gate) are in
**[`V2_1_ARCHITECTURE.md`](V2_1_ARCHITECTURE.md)**.

There were two training generations inside `v2/` itself along the way — `train_v2.py` (early,
now removed) and `train_v2_1.py` (current, tuned hyperparameters, the one that produced the
results below). If you see references to "v2" vs "v2.1" in old commits, `v2.1` is what's live.

---

## 2. Results

Evaluated on the persistent 324-subject holdout (`checkpoints_v2_1/val_split.json`), best
checkpoint = epoch 108:

| | Dice TC | Dice WT | Dice ET | **Dice Mean** | HD95 TC | HD95 WT | HD95 ET | **HD95 Mean (mm)** |
|---|---|---|---|---|---|---|---|---|
| v1 — plain U-Net | 0.8638 | 0.9087 | 0.8503 | 0.8743 | 5.65 | 6.11 | 5.74 | 5.83 |
| **v2.1 — Attention U-Net (deterministic)** | **0.8791** | **0.9206** | **0.8613** | **0.8870** | **4.83** | **4.87** | **5.06** | **4.92** |
| v2.1 — MC Dropout (20 passes) | 0.8794 | 0.9205 | 0.8616 | 0.8872 | 4.81 | 4.88 | 5.04 | 4.91 |

v2.1 improves on **every** metric (+1.3 pts mean Dice, ~16% lower mean HD95), with the biggest
relative gain on ET (0.8503 → 0.8613) — exactly the region attention gates were meant to help.

MC Dropout barely moves the point estimate (+0.0002 mean Dice) — that's expected. Dropout is only
in the bottleneck + two deepest decoder blocks specifically to generate calibrated *uncertainty*,
not to boost accuracy. The payoff is the per-voxel entropy map (§3), not a Dice bump.

Note: training was interrupted at epoch 143/300 (one epoch short of the patience=35
early-stopping trigger) — so `best_model.pth` (epoch 108) is a strong but not perfectly
converged checkpoint. `latest_checkpoint.pth` is resumable via `train_v2_1.py --resume` if you
want to try pushing further.

---

## 3. Uncertainty & calibration (`visualize_uncertainty_v2.py`)

Same MC Dropout idea as v1, ported to the attention model, plus a new calibration analysis this
generation didn't have before.

**Entropy by correctness** (10 example subjects, 20 MC passes):

| Region | Mean Entropy |
|---|---|
| TN (correct background) | 0.0002 |
| TP (correct tumor) | 0.0285 |
| FP (false positive) | 0.1960 |
| FN (false negative) | 0.1112 |

FP entropy is **~6.9× TP entropy** — the model flags its own mistakes with much higher
uncertainty, consistent with v1's ~6× finding. The attention-gate upgrade didn't break this
property.

**Voxel-level calibration** (50 val subjects, brain-mask restricted — new for v2.1):
- **Expected Calibration Error (ECE) = 0.0080** — softmax confidence is well calibrated against
  empirical accuracy.
- **AUROC = 0.8956** — per-voxel entropy alone, with no other signal, is a strong predictor of
  "this voxel is misclassified." In other words: the uncertainty map is directly usable to flag
  low-confidence regions for radiologist review, not just a nice-looking visualization.

Outputs: `uncertainty_vis_v2_1/` (canonical) and `results_summary/` (presentation copy — see §5).

---

## 4. How to reproduce

```bash
# Train (fresh run or --resume from latest_checkpoint.pth)
python train_v2_1.py
python train_v2_1.py --resume

# Evaluate — Dice/HD95 only, deterministic or MC Dropout
python evaluate_v2.py --checkpoint checkpoints_v2_1/best_model.pth --mc_passes 0
python evaluate_v2.py --checkpoint checkpoints_v2_1/best_model.pth --mc_passes 20

# Per-subject figures + aggregate metrics table
python inference_report_v2.py --checkpoint checkpoints_v2_1/best_model.pth \
    --out_dir inference_report_v2_1

# Uncertainty maps + calibration (ECE, AUROC)
python visualize_uncertainty_v2.py --checkpoint checkpoints_v2_1/best_model.pth
python calibration_foreground_only.py --checkpoint checkpoints_v2_1/best_model.pth

# Animated uncertainty GIFs (outputs to uncertainty_gif_v2_1/, gitignored)
python visualize_uncertainty_gif_v2.py --checkpoint checkpoints_v2_1/best_model.pth
```

`checkpoints_v2_1/` (checkpoints + `val_split.json`) is gitignored — too large for git. If
you're starting fresh, `train_v2_1.py` will regenerate `val_split.json` on first run; ping Sahar
for the actual `best_model.pth` if you want the exact epoch-108 checkpoint rather than retraining.

---

## 5. File map

| File / folder | What it is |
|---|---|
| `model_v2.py` | `UNet3DAttn`, `AttentionGate`, `mc_inference()`, `get_region_masks()` |
| `dataset_v2.py` | `BraTSDatasetV2` (augmentation), `get_split_loaders_v2()` (persistent split) |
| `train_v2_1.py` | Training loop, checkpointing, resume, early stopping |
| `evaluate_v2.py` | Fast console-only Dice/HD95 evaluation, deterministic or MC |
| `inference_report_v2.py` | Per-subject figures + aggregate table + MC per-pass stability table |
| `visualize_uncertainty_v2.py` | MC Dropout entropy maps + voxel-level calibration (ECE, AUROC) |
| `calibration_foreground_only.py` | ECE + AUROC recomputed on tumor voxels only (background-excluded) |
| `visualize_uncertainty_gif_v2.py` | Animated 3D / slice-sweep uncertainty GIFs (outputs gitignored) |
| `build_entropy_extremes_examples.py` | Picks lowest/highest-entropy subjects for the paper's example figures |
| `checkpoints_v2_1/` | Checkpoints + persistent val split (gitignored, large) |
| `results_summary/` | Presentation-ready copy of the key figures/tables (see below) |
| `uncertainty_vis_v2_1/`, `inference_report_v2_1/`, `montecarlo_v2_1/` | Canonical (reproducible) script outputs |
| `V2_1_ARCHITECTURE.md` | Full architecture writeup — attention gate mechanics, training setup, per-block param counts |

**`results_summary/`** is worth knowing about specifically: it's a flat, self-contained copy of
the headline figures/tables from `uncertainty_vis_v2_1/`, plus a
`RESULTS_SUMMARY.md` index — meant to be zipped and shared (e.g. for a presentation) without
pulling in the full 324-subject output trees. The canonical, reproducible-from-scripts outputs
live in the other folders; `results_summary/` is a snapshot, regenerate it by hand if the
underlying numbers change.

---

## 6. Where to go next

- Full project history phase-by-phase (both v1 and v2.1): root **`PROGRESS.md`**
- v1 architecture/training/results and literature comparison: root **`CLAUDE.md`**, root
  **`ANALYSIS.md`**
- Attention gate mechanics, per-block param breakdown, full training/inference CLI reference:
  **`V2_1_ARCHITECTURE.md`**