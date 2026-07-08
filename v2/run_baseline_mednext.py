"""
Baseline comparison #2: run a genuine, published MedNeXt checkpoint (Ferreira et al.,
"How we won BraTS 2023 Adult Glioma challenge? Just faking it!", the team that also won
BraTS 2024 — arXiv 2402.17317, weights on Zenodo record 14001262) on our BraTS 2024 val
split, with dropout WE add ourselves (MedNeXt ships with no dropout), and compare
Dice/HD95 + MC-dropout uncertainty against our own v2.1 Attention U-Net.

Checkpoint used: Task241_BraTS_2024_Real, fold 0 (nnUNetTrainerV2_MedNeXt_L_kernel5) —
trained on REAL BraTS 2024 Task 1 (Adult Glioma Post-Treatment) data only, no synthetic
GAN augmentation — the cleanest domain match to our own training data among their releases.
Extracted directly from the 39.8GB Zenodo archive via HTTP range requests (remotezip) —
avoided downloading the full archive (only the ~757MB fold_0 checkpoint + plans.pkl needed).

Architecture reconstructed empirically from the checkpoint's own tensor shapes (confirmed
exact match, zero unexpected/missing keys besides the training-only `dummy_tensor`):
    MedNeXt(in_channels=4, n_channels=32, n_classes=5, exp_r=[3,4,8,8,8,8,8,4,3],
            kernel_size=5, deep_supervision=True, do_res=True, do_res_up_down=True,
            block_counts=[3,4,8,8,8,8,8,4,3], norm_type='group', dim='3d', grn=False)
5 output channels = background(0) + label 1 + label 2 + ET(3) + RC(4). IMPORTANT: their
own dataset.json (`example/dataset_2024_glioma.json` in the Zenodo archive) defines
region membership explicitly as `whole tumor: [1,2,3]`, `tumor core: [2,3]`,
`enhancing tumor: 3` — i.e. their **label 1 = SNFH/edema and label 2 = NCR**, the OPPOSITE
of our own convention (ours: 1=NCR, 2=SNFH). TC must therefore be derived as labels {2,3}
on their raw output, NOT {1,3} (confirmed empirically: using {1,3} produced wildly wrong
TC Dice ~0.28 because it silently substituted edema for necrotic core). WT is unaffected
by the 1/2 swap (symmetric union). RC(4) (resection cavity) has no counterpart in our own
GT — checked empirically whether it should be folded into ET(3) (on the theory that our
own raw-label-4 is old-convention ET, see BASELINE_COMPARISON.md) but predicted RC(4)
voxels overlap our GT at 80-99% with label 2 (edema) and <10% with label 3 (ET) across
sampled subjects, i.e. predicted RC does NOT correspond to our ET class. We instead leave
RC(4) out of all three regions entirely, matching MedNeXt's own official region
definitions above (RC excluded from WT/TC/ET) — `mednext_region_masks()` only tests for
labels 1/2/3 so this falls out for free without an explicit remap step.

Known axis-order pitfall (fixed): our .h5 files store volumes in nibabel's (X,Y,Z) axis
order; MedNeXt/nnU-Net was trained via SimpleITK, whose array convention is (Z,Y,X). Our
crop makes X and Z both 160, so feeding (X,Y,Z) directly is shape-silent but scrambles
left-right vs. superior-inferior structure (verified: WT Dice 0.24 -> 0.66 on one subject,
predicted/GT centroids now match, after transposing spatial axes 0<->2 before inference
and permuting predictions back afterward — see `evaluate()`).

Dropout: MedNeXt has none by default. We append a single nn.Dropout3d to the end of the
8-block bottleneck Sequential (mirrors where our own model and the SegResNet baseline put
theirs) — Dropout has no learnable parameters, so this doesn't disturb the loaded weights
at all (verified: strict-load matches with zero missing/unexpected keys before wrapping).

Outputs v2/baseline_comparison/mednext_results.json, consumed by compare_baseline.py.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric, HausdorffDistanceMetric

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mednext_arch"))
from MedNextV1 import MedNeXt  # noqa: E402

from model_v2 import get_region_masks
from run_baseline_segresnet import RunningStats, get_val_files  # reuse shared helpers

# Our preprocessed .h5 channel order -> MedNeXt's expected [T1C, T1, T2, FLAIR] order
BUNDLE_CHANNEL_ORDER = [1, 0, 2, 3]  # index into [t1n, t1c, t2w, t2f] for [t1c, t1n, t2w, t2f]

REGIONS = ["TC", "WT", "ET"]
PATCH_SIZE = (128, 128, 128)  # from plans.pkl: plans_per_stage[0]['patch_size']


def build_model(checkpoint_path, device, dropout_p=0.2):
    model = MedNeXt(
        in_channels=4, n_channels=32, n_classes=5,
        exp_r=[3, 4, 8, 8, 8, 8, 8, 4, 3], kernel_size=5,
        deep_supervision=True, do_res=True, do_res_up_down=True,
        block_counts=[3, 4, 8, 8, 8, 8, 8, 4, 3], norm_type="group", dim="3d", grn=False,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ckpt["state_dict"].items() if k != "dummy_tensor"}
    result = model.load_state_dict(sd, strict=False)
    assert result.unexpected_keys == [], f"Unexpected keys: {result.unexpected_keys}"
    assert result.missing_keys == ["dummy_tensor"], f"Unexpected missing keys: {result.missing_keys}"

    model.do_ds = False  # only need the full-resolution head for inference

    # Add dropout ourselves — MedNeXt has none. Appending to the Sequential adds no
    # learnable parameters, so this is done AFTER the strict-checked weight load.
    model.bottleneck = nn.Sequential(*list(model.bottleneck.children()), nn.Dropout3d(p=dropout_p))

    return model.to(device)


def normalize_shared_mask(images: np.ndarray) -> np.ndarray:
    """nnU-Net 'nonCT' normalization: per-channel z-score using ONE shared brain mask
    (any channel nonzero), matching `GenericPreprocessor.resample_and_normalize`'s
    `use_nonzero_mask` branch — background left at exactly 0."""
    out = images.copy()
    mask = out.max(axis=0) > 0
    if not mask.any():
        return out
    for c in range(out.shape[0]):
        vals = out[c][mask]
        mean, std = vals.mean(), vals.std()
        out[c][mask] = (vals - mean) / (std + 1e-8)
        out[c][~mask] = 0
    return out


def set_dropout_train(model):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def mednext_region_masks(pred_class: torch.Tensor):
    """Derive TC/WT/ET from a raw MedNeXt prediction (values in {0,1,2,3,4}). Their label
    semantics (dataset_2024_glioma.json): 1=SNFH/edema, 2=NCR, 3=ET — TC is therefore
    {2,3}, NOT {1,3} (our own get_region_masks() assumes our own 1=NCR/2=SNFH convention
    and would silently substitute edema for necrotic core here). RC(4) is excluded from
    all three regions, matching their own official region definitions — predicted RC
    voxels were verified to overlap mostly with our GT's edema, not our ET, so folding
    RC into ET would be wrong (see module docstring)."""
    TC = (pred_class == 2) | (pred_class == 3)
    WT = (pred_class == 1) | (pred_class == 2) | (pred_class == 3)
    ET = (pred_class == 3)
    return TC.long(), WT.long(), ET.long()


def mc_inference_mednext(model, image, inferer, n_passes):
    """MC Dropout inference for the categorical-softmax MedNeXt model.

    Returns:
        mean_pred: (B, 5, H, W, D) mean softmax probability over passes
        entropy:   (B, H, W, D)    categorical predictive entropy of the mean prediction
                   (same formula as our own model — MedNeXt is single-label softmax, unlike
                   the sigmoid multi-label SegResNet baseline)
    """
    model.eval()
    set_dropout_train(model)
    mean_pred = None
    with torch.no_grad():
        for _ in range(n_passes):
            logits = inferer(inputs=image, network=model)
            p = torch.softmax(logits, dim=1)
            mean_pred = p if mean_pred is None else mean_pred + p
    mean_pred = mean_pred / n_passes
    entropy = -(mean_pred * torch.log(mean_pred + 1e-8)).sum(1)
    # `image` was fed in nnU-Net's (Z,Y,X) axis order (see evaluate()) — permute the
    # spatial axes back to our own (X,Y,Z) order before returning.
    mean_pred = mean_pred.permute(0, 1, 4, 3, 2)
    entropy = entropy.permute(0, 3, 2, 1)
    return mean_pred, entropy


def entropy_stats_region(entropy_region: np.ndarray, gt_bin: np.ndarray, pred_bin: np.ndarray):
    gt_bin, pred_bin = gt_bin.astype(bool), pred_bin.astype(bool)
    masks = {
        "TN": (~gt_bin) & (~pred_bin), "TP": (gt_bin) & (pred_bin),
        "FP": (~gt_bin) & (pred_bin), "FN": (gt_bin) & (~pred_bin),
    }
    return {name: entropy_region[mask] for name, mask in masks.items() if mask.any()}


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(args.checkpoint, device, dropout_p=args.dropout_p)
    model.eval()
    print(f"Loaded MedNeXt (Task241_BraTS_2024_Real, fold 0) from {args.checkpoint}")

    inferer = SlidingWindowInferer(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=0.5, mode="gaussian")

    val_files = get_val_files(args.data_dir, args.val_split_json)
    if args.limit:
        val_files = val_files[: args.limit]
    print(f"Evaluating on {len(val_files)} subjects  MC passes: {args.mc_passes}")

    dice_det = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=True)
    hd_det   = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")
    dice_mc  = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=True)
    hd_mc    = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")

    entropy_pool = {r: {"TN": RunningStats(), "TP": RunningStats(), "FP": RunningStats(), "FN": RunningStats()}
                     for r in REGIONS}

    for i, fpath in enumerate(val_files, 1):
        with h5py.File(fpath, "r") as f:
            images_np = f["images"][:].astype(np.float32)  # (4, H, W, D) order: t1n,t1c,t2w,t2f
            seg_np    = f["seg"][:].astype(np.int64)

        images_reordered = images_np[BUNDLE_CHANNEL_ORDER]  # -> [t1c, t1n, t2w, t2f]
        images_norm = normalize_shared_mask(images_reordered)
        # Our .h5 stores volumes in nibabel's (X,Y,Z) axis order; nnU-Net/MedNeXt was
        # trained on SimpleITK's (Z,Y,X) array order. X and Z are both 160 after our
        # crop, so feeding (X,Y,Z) directly is shape-silent but scrambles left-right vs
        # superior-inferior structure (verified empirically: WT Dice 0.24 -> 0.66 on a
        # single subject after fixing this, with predicted/GT centroids now matching).
        # Transpose spatial axes 0<->2 here; predictions are permuted back afterward.
        images_norm = np.ascontiguousarray(images_norm.transpose(0, 3, 2, 1))
        images_t = torch.from_numpy(images_norm).unsqueeze(0).to(device)

        seg_t = torch.from_numpy(seg_np).unsqueeze(0).to(device)
        TC_gt, WT_gt, ET_gt = get_region_masks(seg_t)
        gt_r = torch.stack([TC_gt, WT_gt, ET_gt], dim=1).float()

        # Deterministic (eval mode, no dropout)
        with torch.no_grad():
            det_logits = inferer(inputs=images_t, network=model)
            det_logits = det_logits.permute(0, 1, 4, 3, 2)  # back to (B,C,X,Y,Z)
            det_pred_class = det_logits.argmax(dim=1)
        TC_p, WT_p, ET_p = mednext_region_masks(det_pred_class)
        det_pred_r = torch.stack([TC_p, WT_p, ET_p], dim=1).float()
        dice_det(y_pred=det_pred_r, y=gt_r)
        hd_det(y_pred=det_pred_r, y=gt_r)

        # MC Dropout
        if args.mc_passes > 0:
            mean_pred, entropy = mc_inference_mednext(model, images_t, inferer, args.mc_passes)
            mc_pred_class = mean_pred.argmax(dim=1)
            TC_m, WT_m, ET_m = mednext_region_masks(mc_pred_class)
            mc_pred_r = torch.stack([TC_m, WT_m, ET_m], dim=1).float()
            dice_mc(y_pred=mc_pred_r, y=gt_r)
            hd_mc(y_pred=mc_pred_r, y=gt_r)

            entropy_np = entropy[0].cpu().numpy()
            pred_r_np = mc_pred_r[0].cpu().numpy()
            gt_r_np = gt_r[0].cpu().numpy()
            for r_i, region in enumerate(REGIONS):
                stats = entropy_stats_region(entropy_np, gt_r_np[r_i], pred_r_np[r_i])
                for name, vals in stats.items():
                    entropy_pool[region][name].update(vals)

        if i % 25 == 0 or i == len(val_files):
            print(f"  {i}/{len(val_files)}")

    def summarize(metric):
        arr = metric.aggregate().cpu().numpy()
        return {"TC": float(arr[0]), "WT": float(arr[1]), "ET": float(arr[2]),
                "mean": float(np.nanmean(np.where(np.isfinite(arr), arr, np.nan)))}

    results = {
        "model": "mednext_baseline_brats2024_real_fold0",
        "checkpoint": args.checkpoint,
        "n_subjects": len(val_files),
        "mc_passes": args.mc_passes,
        "dice_det": summarize(dice_det),
        "hd95_det": summarize(hd_det),
    }
    if args.mc_passes > 0:
        results["dice_mc"] = summarize(dice_mc)
        results["hd95_mc"] = summarize(hd_mc)
        entropy_summary = {}
        for region in REGIONS:
            entropy_summary[region] = {}
            for name, running in entropy_pool[region].items():
                finalized = running.finalize()
                if finalized is not None:
                    entropy_summary[region][name] = finalized
        results["entropy_by_region"] = entropy_summary

    print("\n=== MedNeXt Baseline Results ===")
    print("Deterministic Dice:", results["dice_det"])
    print("Deterministic HD95:", results["hd95_det"])
    if args.mc_passes > 0:
        print("MC-20 Dice:", results["dice_mc"])
        print("MC-20 HD95:", results["hd95_mc"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mednext_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Run published MedNeXt (Ferreira et al.) BraTS baseline")
    p.add_argument("--checkpoint", default="external_models/mednext_task241_fold0/model_best.model")
    p.add_argument("--data_dir", default="/home/sahar/CV_medical_data_project/processed/train")
    p.add_argument("--val_split_json", default="checkpoints_v2_1/val_split.json")
    p.add_argument("--mc_passes", type=int, default=20)
    p.add_argument("--dropout_p", type=float, default=0.2)
    p.add_argument("--limit", type=int, default=None, help="Cap number of subjects (smoke testing)")
    p.add_argument("--out_dir", default="baseline_comparison")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
