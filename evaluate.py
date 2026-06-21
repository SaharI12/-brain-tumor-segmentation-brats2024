import argparse
import os
import random
from glob import glob

import numpy as np

import torch
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from dataset import BraTSDataset
from model import UNet3D, mc_inference, get_region_masks


def get_val_files(data_dir, val_split, seed):
    all_files = sorted(glob(os.path.join(data_dir, "*.h5")))
    rng = random.Random(seed)
    rng.shuffle(all_files)
    n_val = int(len(all_files) * val_split)
    return all_files[:n_val]


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt      = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = UNet3D(
        in_channels=4,
        out_channels=4,
        init_features=ckpt_args.get("init_features", args.init_features),
        dropout_p=ckpt_args.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"Checkpoint: epoch {ckpt.get('epoch', '?')}  "
        f"best Dice {ckpt.get('best_dice', float('nan')):.4f}"
    )

    val_files = get_val_files(
        data_dir=args.data_dir,
        val_split=ckpt_args.get("val_split", args.val_split),
        seed=ckpt_args.get("seed", args.seed),
    )
    val_loader = DataLoader(
        BraTSDataset(val_files, augment=False),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
    )
    print(f"Evaluating on {len(val_files)} subjects  MC passes: {args.mc_passes}")

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=True)
    hd_metric   = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")

    model.eval()

    for i, (images, seg) in enumerate(val_loader):
        images, seg = images.to(device), seg.to(device)

        if args.mc_passes > 0:
            mean_pred, _ = mc_inference(model, images, n_passes=args.mc_passes)
            pred_class   = mean_pred.argmax(dim=1)
        else:
            with torch.no_grad():
                pred_class = model(images).argmax(dim=1)

        TC_pred, WT_pred, ET_pred = get_region_masks(pred_class)
        TC_gt,   WT_gt,   ET_gt   = get_region_masks(seg)

        pred_r = torch.stack([TC_pred, WT_pred, ET_pred], dim=1).float()
        gt_r   = torch.stack([TC_gt,   WT_gt,   ET_gt],   dim=1).float()

        dice_metric(y_pred=pred_r, y=gt_r)
        hd_metric(y_pred=pred_r, y=gt_r)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(val_loader)}")

    dice_scores = dice_metric.aggregate().cpu().numpy()
    hd_scores   = hd_metric.aggregate().cpu().numpy()
    # HausdorffDistanceMetric returns NaN for subjects/regions where both pred and gt
    # are empty (e.g., no ET in post-treatment cases). Use nanmean to exclude them.
    hd_scores_safe = np.where(np.isfinite(hd_scores), hd_scores, np.nan)

    print("\n=== Evaluation Results ===")
    print(f"       {'TC':>7}  {'WT':>7}  {'ET':>7}  {'Mean':>7}")
    print(f"Dice:  {dice_scores[0]:7.4f}  {dice_scores[1]:7.4f}  {dice_scores[2]:7.4f}  {float(np.nanmean(dice_scores)):7.4f}")
    print(f"HD95:  {hd_scores_safe[0]:7.2f}  {hd_scores_safe[1]:7.2f}  {hd_scores_safe[2]:7.2f}  {float(np.nanmean(hd_scores_safe)):7.2f}")


def parse_args():
    p = argparse.ArgumentParser(description="BraTS 2024 Evaluation")
    p.add_argument("--checkpoint",    required=True, help="Path to best_model.pth")
    p.add_argument("--data_dir",      default="processed/train")
    p.add_argument("--val_split",     type=float, default=0.2)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--init_features", type=int,   default=16)
    p.add_argument("--mc_passes",     type=int,   default=0,
                   help="MC Dropout inference passes (0 = deterministic)")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())