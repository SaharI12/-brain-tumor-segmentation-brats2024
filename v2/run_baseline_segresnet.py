"""
Baseline comparison: run MONAI's published `brats_mri_segmentation` bundle
(SegResNet, Myronenko 2018, trained on BraTS 2018) on our BraTS 2024 val split,
with and without MC Dropout, and compare Dice/HD95 + uncertainty behavior
against our own v2.1 Attention U-Net.

Bundle: https://huggingface.co/MONAI/brats_mri_segmentation (Apache-2.0)
  - Architecture: SegResNet(blocks_down=[1,2,2,4], blocks_up=[1,1,1], init_filters=16,
    in_channels=4, out_channels=3, dropout_prob=0.2) — confirmed from configs/train.json
  - Input channel order: [T1c, T1, T2, FLAIR]  (channel_def in configs/metadata.json)
  - Output: 3-channel sigmoid [TC, WT, ET] — already our exact region format
  - Preprocessing: NormalizeIntensityd(nonzero=True, channel_wise=True) — i.e. per-channel
    z-score over nonzero voxels only, background left at 0. This is scale-invariant to the
    positive linear rescaling our own preprocessing already applied (dividing by a global
    max), so it can be applied directly to our preprocessed [0,1] volumes without needing
    raw NIfTI files.
  - Domain shift caveat: this model has never seen BraTS 2024 data. See
    v2/BASELINE_COMPARISON.md for full discussion.

Outputs v2/baseline_comparison/segresnet_results.json (Dice/HD95 + per-region entropy stats),
consumed by compare_baseline.py.
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.nets import SegResNet

from model_v2 import get_region_masks

# Our preprocessed .h5 channel order -> bundle's expected [T1c, T1, T2, FLAIR] order
OUR_CHANNELS = ["t1n", "t1c", "t2w", "t2f"]
BUNDLE_CHANNEL_ORDER = [1, 0, 2, 3]  # index into OUR_CHANNELS for [t1c, t1n, t2w, t2f]

REGIONS = ["TC", "WT", "ET"]


def get_val_files(data_dir, split_path):
    with open(split_path) as f:
        saved = json.load(f)
    return [os.path.join(data_dir, fn) for fn in saved["val"]]


def build_model(bundle_dir, device):
    model = SegResNet(
        blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16,
        in_channels=4, out_channels=3, dropout_prob=0.2,
    ).to(device)
    state_dict = torch.load(os.path.join(bundle_dir, "models", "model.pt"), map_location=device)
    model.load_state_dict(state_dict)
    return model


def normalize_nonzero_channelwise(images: np.ndarray) -> np.ndarray:
    """Per-channel z-score over nonzero voxels, background left at 0.
    Equivalent to MONAI's NormalizeIntensityd(nonzero=True, channel_wise=True)."""
    out = images.copy()
    for c in range(out.shape[0]):
        mask = out[c] != 0
        if not mask.any():
            continue
        mean = out[c][mask].mean()
        std = out[c][mask].std()
        if std == 0:
            continue
        out[c][mask] = (out[c][mask] - mean) / std
    return out


def set_dropout_train(model):
    """Force only Dropout submodules into train() mode; everything else (GroupNorm etc.)
    stays in eval() mode, matching this bundle's norm layers (no running-stat dependence)."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def mc_inference_segresnet(model, image, inferer, n_passes):
    """MC Dropout inference for the sigmoid multi-label SegResNet baseline.

    Returns:
        mean_pred: (B, 3, H, W, D) mean sigmoid probability over passes, channels [TC, WT, ET]
        entropy:   (B, 3, H, W, D) per-channel binary entropy of the mean prediction
                   H = -p*log(p) - (1-p)*log(1-p)  (independent per sigmoid channel,
                   distinct from our own model's categorical softmax entropy)
    """
    model.eval()
    set_dropout_train(model)
    mean_pred = None
    with torch.no_grad():
        for _ in range(n_passes):
            logits = inferer(inputs=image, network=model)
            p = torch.sigmoid(logits)
            mean_pred = p if mean_pred is None else mean_pred + p
    mean_pred = mean_pred / n_passes
    entropy = -(mean_pred * torch.log(mean_pred + 1e-8) +
                (1 - mean_pred) * torch.log(1 - mean_pred + 1e-8))
    return mean_pred, entropy


def entropy_stats_region(entropy_region: np.ndarray, gt_bin: np.ndarray, pred_bin: np.ndarray):
    gt_bin = gt_bin.astype(bool)
    pred_bin = pred_bin.astype(bool)
    masks = {
        "TN": (~gt_bin) & (~pred_bin),
        "TP": (gt_bin) & (pred_bin),
        "FP": (~gt_bin) & (pred_bin),
        "FN": (gt_bin) & (~pred_bin),
    }
    return {name: entropy_region[mask] for name, mask in masks.items() if mask.any()}


class RunningStats:
    """Online mean/std accumulation — avoids holding all-subject voxel arrays in memory."""

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0

    def update(self, values: np.ndarray):
        self.count += values.size
        self.total += float(values.sum())
        self.total_sq += float((values ** 2).sum())

    def finalize(self):
        if self.count == 0:
            return None
        mean = self.total / self.count
        var = max(self.total_sq / self.count - mean ** 2, 0.0)
        return {"mean": mean, "std": float(np.sqrt(var)), "count": self.count}


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(args.bundle_dir, device)
    model.eval()
    print(f"Loaded SegResNet baseline from {args.bundle_dir}")

    inferer = SlidingWindowInferer(roi_size=(240, 240, 160), sw_batch_size=1, overlap=0.5)

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
        images_norm = normalize_nonzero_channelwise(images_reordered)
        images_t = torch.from_numpy(images_norm).unsqueeze(0).to(device)

        seg_t = torch.from_numpy(seg_np).unsqueeze(0).to(device)
        TC_gt, WT_gt, ET_gt = get_region_masks(seg_t)
        gt_r = torch.stack([TC_gt, WT_gt, ET_gt], dim=1).float()

        # Deterministic (eval mode, no dropout)
        with torch.no_grad():
            det_logits = inferer(inputs=images_t, network=model)
            det_pred = (torch.sigmoid(det_logits) > 0.5).float()
        dice_det(y_pred=det_pred, y=gt_r)
        hd_det(y_pred=det_pred, y=gt_r)

        # MC Dropout
        if args.mc_passes > 0:
            mean_pred, entropy = mc_inference_segresnet(model, images_t, inferer, args.mc_passes)
            mc_pred = (mean_pred > 0.5).float()
            dice_mc(y_pred=mc_pred, y=gt_r)
            hd_mc(y_pred=mc_pred, y=gt_r)

            mean_pred_np = mean_pred[0].cpu().numpy()
            entropy_np   = entropy[0].cpu().numpy()
            pred_np      = mc_pred[0].cpu().numpy()
            gt_np        = gt_r[0].cpu().numpy()
            for r_i, region in enumerate(REGIONS):
                stats = entropy_stats_region(entropy_np[r_i], gt_np[r_i], pred_np[r_i])
                for name, vals in stats.items():
                    entropy_pool[region][name].update(vals)

        if i % 25 == 0 or i == len(val_files):
            print(f"  {i}/{len(val_files)}")

    def summarize(metric):
        arr = metric.aggregate().cpu().numpy()
        return {"TC": float(arr[0]), "WT": float(arr[1]), "ET": float(arr[2]),
                "mean": float(np.nanmean(np.where(np.isfinite(arr), arr, np.nan)))}

    results = {
        "model": "segresnet_baseline_brats2018",
        "bundle_dir": args.bundle_dir,
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

    print("\n=== SegResNet Baseline Results ===")
    print("Deterministic Dice:", results["dice_det"])
    print("Deterministic HD95:", results["hd95_det"])
    if args.mc_passes > 0:
        print("MC-20 Dice:", results["dice_mc"])
        print("MC-20 HD95:", results["hd95_mc"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "segresnet_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Run MONAI SegResNet BraTS baseline for MC comparison")
    p.add_argument("--bundle_dir", default="external_models/brats_mri_segmentation")
    p.add_argument("--data_dir", default="/home/sahar/CV_medical_data_project/processed/train")
    p.add_argument("--val_split_json", default="checkpoints_v2_1/val_split.json")
    p.add_argument("--mc_passes", type=int, default=20)
    p.add_argument("--limit", type=int, default=None, help="Cap number of subjects (smoke testing)")
    p.add_argument("--out_dir", default="baseline_comparison")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
