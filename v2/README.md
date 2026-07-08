# v2 — Attention U-Net (BraTS 2024 GLI)

This folder is the second generation of the project's segmentation model. If you're picking this
up for the first time, read this file top to bottom — it's the map. The deep-dive docs
(architecture mechanics, baseline methodology) are linked inline rather than repeated here.

**Start here if you only have five minutes:** §1 (what changed and why) and §2 (results table).

---

## 0. How this relates to the root project

The root of the repo (`data_playing.py`, `preprocess.py`, `model.py`, `train.py`, ...) is **v1** —
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

Outputs: `uncertainty_vis_v2_1/` (canonical) and `results_summary/` (presentation copy — see §6).

---

## 4. How our model compares to published, pretrained BraTS models

Two published checkpoints were downloaded and run through an adapter pipeline (channel reorder,
normalization, sliding-window inference, MC Dropout) on **our own identical 324-subject val
split** — a true head-to-head, not a literature-numbers comparison.

| Model — Mode | Dice Mean | HD95 Mean (mm) |
|---|---|---|
| **v2.1 (ours) — Deterministic** | **0.8870** | **4.92** |
| v2.1 (ours) — MC Dropout (20) | 0.8869 | 4.92 |
| SegResNet (MONAI bundle, trained on BraTS 2018) — Deterministic | 0.5791 | 18.09 |
| SegResNet — MC Dropout (20) | 0.5860 | 17.13 |
| MedNeXt (Ferreira et al., BraTS 2023/2024 challenge winner) | *run in progress — see below* | |

**SegResNet** (MONAI's published `brats_mri_segmentation` bundle) scores much lower on our data
(0.58 vs 0.887 mean Dice) — expected: it never saw BraTS 2024's post-treatment scans (surgical
cavities, radiation effects, pseudoprogression). Its MC-dropout uncertainty is also much blunter
than ours (FP/TP entropy ratio 2.6× vs. our 6.9×) — a model trained on your actual domain gives
you a sharper uncertainty signal, not just better raw accuracy.

**MedNeXt** (the actual BraTS-winning architecture, trained on real BraTS 2024 data — the closest
domain match of any public checkpoint we found) is being evaluated now
(`run_baseline_mednext.py`, n=324, 5 MC passes, background run). Two real bugs had to be fixed
before the numbers were trustworthy — worth knowing about since they're the kind of thing that
silently corrupts results without ever raising an error:

1. **Axis order.** Our `.h5` files store volumes in nibabel's `(X,Y,Z)` order; MedNeXt was trained
   via SimpleITK, whose array convention is `(Z,Y,X)`. Our crop happens to make X and Z both 160,
   so feeding the wrong order in is *shape-compatible* — no crash, no obvious symptom — but it
   scrambles left-right vs. superior-inferior structure. Caught via a centroid check
   (predicted-vs-GT tumor centroid was off by ~30 voxels on one axis); fixed by transposing spatial
   axes before inference and back afterward. WT Dice on a test subject went 0.24 → 0.66.
2. **Label semantics.** MedNeXt outputs a 5th class, RC (resection cavity), which our own GT
   doesn't have. The first draft of the script merged RC into our ET class on an unverified
   assumption. Checked empirically instead: predicted RC voxels overlap 80–99% with our GT's
   *edema* label, not ET. Now RC is excluded from all three regions entirely, matching MedNeXt's
   own official region definitions.

Full methodology and caveats for both baselines: **[`BASELINE_COMPARISON.md`](BASELINE_COMPARISON.md)**.

---

## 5. How to reproduce

```bash
# Train (fresh run or --resume from latest_checkpoint.pth)
python train_v2_1.py
python train_v2_1.py --resume

# Evaluate — Dice/HD95 only, deterministic or MC Dropout
python evaluate_v2.py --checkpoint checkpoints_v2_1/best_model.pth --mc_passes 0
python evaluate_v2.py --checkpoint checkpoints_v2_1/best_model.pth --mc_passes 20

# Per-subject figures + aggregate metrics table
python inference_report_v2.py --checkpoint checkpoints_v2_1/best_model.pth \
    --out_dir exploration_output/inference_report_v2_1

# Uncertainty maps + calibration
python visualize_uncertainty_v2.py --checkpoint checkpoints_v2_1/best_model.pth

# Published-baseline comparisons
python run_baseline_segresnet.py
python run_baseline_mednext.py --mc_passes 20   # slow — sliding-window inference, ~10-12hr at full scale
python compare_baseline.py
```

`checkpoints_v2_1/` (checkpoints + `val_split.json`) and `external_models/` (downloaded
third-party weights) are gitignored — too large for git, and reproducible/re-downloadable. If
you're starting fresh, `train_v2_1.py` will regenerate `val_split.json` on first run; ping Sahar
for the actual `best_model.pth` if you want the exact epoch-108 checkpoint rather than retraining.

---

## 6. File map

| File / folder | What it is |
|---|---|
| `model_v2.py` | `UNet3DAttn`, `AttentionGate`, `mc_inference()`, `get_region_masks()` |
| `dataset_v2.py` | `BraTSDatasetV2` (augmentation), `get_split_loaders_v2()` (persistent split) |
| `train_v2_1.py` | Training loop, checkpointing, resume, early stopping |
| `evaluate_v2.py` | Fast console-only Dice/HD95 evaluation, deterministic or MC |
| `inference_report_v2.py` | Per-subject figures + aggregate table + MC per-pass stability table |
| `visualize_uncertainty_v2.py` | MC Dropout entropy maps + voxel-level calibration (ECE, AUROC) |
| `run_baseline_segresnet.py` | Runs MONAI's published SegResNet bundle (+ MC Dropout) on our val split |
| `run_baseline_mednext.py` | Runs the published MedNeXt checkpoint (+ MC Dropout) on our val split |
| `mednext_arch/` | MedNeXt architecture source (third-party, needed to load the checkpoint) |
| `compare_baseline.py` | Assembles the v2.1 / SegResNet / MedNeXt comparison figures |
| `checkpoints_v2_1/` | Checkpoints + persistent val split (gitignored, large) |
| `external_models/` | Downloaded third-party pretrained weights (gitignored, large) |
| `results_summary/` | Presentation-ready copy of the key figures/tables (see below) |
| `uncertainty_vis_v2_1/`, `inference_report_v2_1/`, `montecarlo_v2_1/`, `baseline_comparison/` | Canonical (reproducible) script outputs |
| `V2_1_ARCHITECTURE.md` | Full architecture writeup — attention gate mechanics, training setup, per-block param counts |
| `BASELINE_COMPARISON.md` | Full methodology + caveats for the SegResNet/MedNeXt comparisons |

**`results_summary/`** is worth knowing about specifically: it's a flat, self-contained copy of
the headline figures/tables from `uncertainty_vis_v2_1/` and `baseline_comparison/`, plus a
`RESULTS_SUMMARY.md` index — meant to be zipped and shared (e.g. for a presentation) without
pulling in the full 324-subject output trees. The canonical, reproducible-from-scripts outputs
live in the other folders; `results_summary/` is a snapshot, regenerate it by hand if the
underlying numbers change.

---

## 7. Where to go next

- Full project history phase-by-phase (both v1 and v2.1): root **`PROGRESS.md`**
- v1 architecture/training/results and literature comparison: root **`CLAUDE.md`**, root
  **`ANALYSIS.md`**
- Attention gate mechanics, per-block param breakdown, full training/inference CLI reference:
  **`V2_1_ARCHITECTURE.md`**
- Baseline comparison methodology, caveats, entropy-formula differences between models:
  **`BASELINE_COMPARISON.md`**