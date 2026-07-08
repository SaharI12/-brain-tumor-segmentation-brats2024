import argparse
import json
import os

import numpy as np

import torch
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from dataset_v2 import BraTSDatasetV2
from model_v2 import UNet3DAttn, mc_inference, get_region_masks


def get_val_files(data_dir, split_path):
    with open(split_path) as f:
        saved = json.load(f)
    return [os.path.join(data_dir, fn) for fn in saved["val"]]


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt      = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = UNet3DAttn(
        in_channels=4,
        out_channels=4,
        init_features=ckpt_args.get("init_features", args.init_features),
        dropout_p=ckpt_args.get("dropout", 0.15),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"Checkpoint: epoch {ckpt.get('epoch', '?')}  "
        f"best Dice {ckpt.get('best_dice', float('nan')):.4f}"
    )

    split_path = args.val_split_json or os.path.join(os.path.dirname(args.checkpoint), "val_split.json")
    val_files = get_val_files(args.data_dir, split_path)
    val_loader = DataLoader(
        BraTSDatasetV2(val_files, augment=False),
        batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
    )
    print(f"Evaluating on {len(val_files)} subjects (split: {split_path})  MC passes: {args.mc_passes}")

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

    if args.json_out:
        results = {
            "model": "v2.1_attention_unet",
            "checkpoint": args.checkpoint,
            "n_subjects": len(val_files),
            "mc_passes": args.mc_passes,
            ("dice_mc" if args.mc_passes > 0 else "dice_det"): {
                "TC": float(dice_scores[0]), "WT": float(dice_scores[1]),
                "ET": float(dice_scores[2]), "mean": float(np.nanmean(dice_scores)),
            },
            ("hd95_mc" if args.mc_passes > 0 else "hd95_det"): {
                "TC": float(hd_scores_safe[0]), "WT": float(hd_scores_safe[1]),
                "ET": float(hd_scores_safe[2]), "mean": float(np.nanmean(hd_scores_safe)),
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nSaved: {args.json_out}")


def parse_args():
    p = argparse.ArgumentParser(description="BraTS 2024 v2.1 (Attention U-Net) Evaluation")
    p.add_argument("--checkpoint",     required=True, help="Path to best_model.pth / latest_checkpoint.pth")
    p.add_argument("--data_dir",       default="/home/sahar/CV_medical_data_project/processed/train")
    p.add_argument("--val_split_json", default=None,
                   help="Path to val_split.json (default: alongside checkpoint)")
    p.add_argument("--init_features",  type=int, default=32)
    p.add_argument("--mc_passes",      type=int, default=0,
                   help="MC Dropout inference passes (0 = deterministic)")
    p.add_argument("--json_out",       default=None,
                   help="Optional path to also dump aggregate Dice/HD95 as JSON")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())