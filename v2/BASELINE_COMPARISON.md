# Baseline Comparison — v2.1 vs. a Published BraTS Model

## Overview — what, why, how

**What:** we compare our v2.1 Attention U-Net against a genuinely different, **published, publicly
downloadable** BraTS segmentation model — not another literature Dice table (see `ANALYSIS.md` §7 for
that), but an actual model we downloaded, ran with our own preprocessing/evaluation pipeline, and
subjected to the same Monte Carlo Dropout uncertainty analysis as our own model.

**Why:** the existing literature comparison (`ANALYSIS.md` §7) compares numbers copied from papers,
evaluated on different held-out sets with different metric definitions (lesion-wise vs scan-level) —
it says nothing about whether another model's *uncertainty* behaves the way ours does. To get a real
MC-dropout comparison we need a model that (a) we can actually run, and (b) genuinely has dropout
layers active during training, so its released weights support MC inference without retraining it
ourselves (which would no longer be "someone else's published model").

**How:** of the publicly downloadable pretrained BraTS checkpoints found (MONAI `brats_mri_segmentation`,
SwinUNETR BraTS21, nnU-Net BraTS21), only **MONAI's `brats_mri_segmentation`** (SegResNet, Myronenko
2018) was trained with `dropout_prob > 0`. SwinUNETR BraTS21 and nnU-Net BraTS21 are both released with
`dropout_rate=0.0` / no dropout layers — architecturally incapable of MC Dropout without retraining. So
SegResNet is the baseline used here, run through an adapter pipeline described below, evaluated on
**the same 324-subject persistent val split** used for v2.1.

---

## 1. The baseline model

**Source:** [`MONAI/brats_mri_segmentation`](https://huggingface.co/MONAI/brats_mri_segmentation)
(Hugging Face, Apache-2.0). Downloaded via:
```
python -m monai.bundle download --name brats_mri_segmentation --bundle_dir v2/external_models/
```

| | Baseline (SegResNet) | Ours (v2.1, UNet3DAttn) |
|---|---|---|
| Architecture | SegResNet (Myronenko 2018) | Attention U-Net |
| Params | 4.7M (`init_filters=16`) | 21.8M (`init_features=32`) |
| Dropout | `dropout_prob=0.2`, one `Dropout3d` at the bottleneck | dropout in bottleneck + 2 deepest decoder blocks |
| Output | 3-channel **sigmoid**, independent [TC, WT, ET] | 4-channel **softmax**, {bg, NCR, SNFH, ET} → regions derived |
| Trained on | **BraTS 2018** (285 cases, pre-treatment) | **BraTS 2024 GLI** (1,621 cases, post-treatment) |
| Reported val Dice (their own eval) | 0.8518 mean (TC 0.8559 / WT 0.9026 / ET 0.7905) | 0.8870 mean (this project, §12 of `PROGRESS.md`) |

Confirmed directly from the bundle's `configs/train.json`:
`SegResNet(blocks_down=[1,2,2,4], blocks_up=[1,1,1], init_filters=16, in_channels=4, out_channels=3, dropout_prob=0.2)`.

## 2. Adapter pipeline (`v2/run_baseline_segresnet.py`)

The baseline was trained on a different (older, pre-treatment) dataset with its own channel order and
normalization convention, so a small adapter is needed to run it on our preprocessed `.h5` files —
**no relabeling of predictions was needed**, since its output channels are already [TC, WT, ET]:

1. **Channel reorder** — our `.h5` channel order is `[t1n, t1c, t2w, t2f]`; the bundle's
   `channel_def` expects `[T1c, T1, T2, FLAIR]`. We reorder via index `[1, 0, 2, 3]`.
2. **Normalization** — the bundle's own preprocessing is
   `NormalizeIntensityd(nonzero=True, channel_wise=True)`: per-channel z-score computed over nonzero
   voxels only, background left at exactly 0. This is applied directly to our already
   global-max-normalized `[0,1]` volumes rather than to raw NIfTI — z-score is invariant to the
   positive linear rescaling our own preprocessing already applied, so the result is identical to
   normalizing from raw intensities.
3. **Inference** — `monai.inferers.SlidingWindowInferer(roi_size=[240,240,160], overlap=0.5)`, the
   bundle's own trained window size. Our volumes (160×208×160) are smaller in every dimension; MONAI's
   inferer pads internally, confirmed working during smoke testing (no shape errors).
4. **MC Dropout** — `model.eval()` (freezes GroupNorm, which has no running stats and is unaffected by
   train/eval mode anyway) then explicitly `.train()` on only the `Dropout3d` submodule for each of 20
   stochastic passes, mirroring our own `mc_inference()` but restricted to the actual dropout layer
   rather than the whole model.
5. **Uncertainty formula** — this model's output is 3 *independent* sigmoid channels (a voxel can be
   TC=1 and WT=0 simultaneously, unlike our mutually-exclusive softmax), so entropy is **binary
   entropy per channel**: `H = -p·log(p) - (1-p)·log(1-p)`, computed separately for TC/WT/ET — distinct
   from our single categorical entropy over the 4-way softmax. The two are not numerically the same
   quantity; comparisons below use our whole-tumor-binary entropy against the baseline's WT-channel
   entropy as the closest analog.

## 3. Results — Dice / HD95 (same 324-subject val split for v2.1 and the baseline)

| Model — Mode | Dice TC | Dice WT | Dice ET | Dice Mean | HD95 TC | HD95 WT | HD95 ET | HD95 Mean |
|---|---|---|---|---|---|---|---|---|
| **v2.1 (ours) — Deterministic** | **0.8791** | **0.9206** | **0.8613** | **0.8870** | **4.83** | **4.87** | **5.06** | **4.92** |
| v2.1 (ours) — MC Dropout (20) | 0.8791 | 0.9205 | 0.8612 | 0.8869 | 4.84 | 4.85 | 5.07 | 4.92 |
| SegResNet (published) — Deterministic | 0.5620 | 0.7797 | 0.3955 | 0.5791 | 19.15 | 15.31 | 19.82 | 18.09 |
| SegResNet (published) — MC Dropout (20) | 0.5723 | 0.7896 | 0.3961 | 0.5860 | 18.21 | 14.07 | 19.10 | 17.13 |

Figure: `v2/baseline_comparison/comparison_table.png`

**The published baseline substantially underperforms our model on our data** — mean Dice
0.58 vs our 0.887 (a ~0.31 absolute / ~53% relative gap), mean HD95 ~17–18mm vs our ~4.9mm (roughly
3.5× worse). **ET is hit hardest** (0.396 vs 0.861) and **TC second hardest** (0.562–0.572 vs 0.879),
while **WT degrades least** (0.780–0.790 vs 0.921). This ordering makes sense under the domain-shift
explanation below: WT is a coarse boundary (core + edema) that generalizes better across cohorts, while
TC and especially ET depend on fine, treatment-sensitive intensity patterns.

**Why the gap is expected, not a flaw in the baseline:** this model has never seen a single BraTS 2024
scan. BraTS 2024 GLI is **post-treatment** — surgical resection cavities, radiation effects, and
pseudoprogression are absent from BraTS 2018's pre-treatment cohort the baseline was trained on. This
is a zero-shot cross-domain evaluation, not a controlled architecture ablation; see caveats below.

## 4. Results — MC Dropout Uncertainty

| Region (whole-tumor binary) | TN | TP | FP | FN |
|---|---|---|---|---|
| **Ours (v2.1)** | 0.0002 | 0.0285 | 0.1960 | 0.1112 |
| SegResNet baseline (WT channel) | 0.0136 | 0.1725 | 0.4507 | 0.3656 |

Figure: `v2/baseline_comparison/entropy_comparison.png`

Both models show the expected qualitative ordering (TN < TP < FN ≤ FP) — **the baseline's dropout-based
uncertainty transfers qualitatively to a new domain even though its predictions don't**. But the
*separation* between correct and incorrect predictions is much weaker for the baseline:

- Ours: FP entropy is **~6.9×** TP entropy (0.1960 / 0.0285)
- Baseline: FP entropy is only **~2.6×** TP entropy (0.4507 / 0.1725)

The baseline's TP entropy (0.173) is already almost as high as its own FN entropy (0.366) — i.e. even
when it's *right*, it's not very confident, so its entropy map is a much blunter signal for flagging
unreliable predictions than ours. Full per-region (TC/WT/ET) entropy tables for the baseline are in
`v2/baseline_comparison/segresnet_results.json`.

**Interpretation:** MC Dropout uncertainty is not a magic property of dropout alone — a model trained
on the actual target domain (ours) produces uncertainty estimates that discriminate errors far more
sharply than a model whose dropout was tuned for a different data distribution (the baseline). This is
a genuinely new finding this comparison surfaces, not something the Dice numbers alone would show.

## 5. Caveats

1. **Domain shift, not a fair architecture ablation.** The baseline was trained on BraTS 2018
   (pre-treatment, 285 cases); ours on BraTS 2024 GLI (post-treatment, 1,621 cases). The Dice gap
   mostly reflects this shift, not SegResNet vs. Attention U-Net as architectures in a vacuum.
2. **No fine-tuning was performed on either side** — the baseline runs fully zero-shot, by design (the
   point was to test a genuinely *published, unmodified* model).
3. **Structurally different uncertainty representations** — softmax categorical entropy (ours) vs.
   per-channel sigmoid binary entropy (baseline) are not the same quantity; the WT-channel comparison
   above is the closest fair analog, not an exact equivalence.
4. Only one dropout layer exists in the baseline (bottleneck `Dropout3d`, vs. three dropout sites in
   ours) — less stochastic depth could also contribute to its blunter uncertainty separation, not just
   domain shift.

## Reproduce

```bash
cd v2
python -m monai.bundle download --name brats_mri_segmentation --bundle_dir external_models/
python run_baseline_segresnet.py --mc_passes 20                       # -> baseline_comparison/segresnet_results.json
python evaluate_v2.py --checkpoint checkpoints_v2_1/best_model.pth --mc_passes 0  --json_out baseline_comparison/v21_det.json
python evaluate_v2.py --checkpoint checkpoints_v2_1/best_model.pth --mc_passes 20 --json_out baseline_comparison/v21_mc20.json
python compare_baseline.py                                            # -> baseline_comparison/{comparison_table,entropy_comparison}.png
```
