"""
Find the val subject with the highest and the lowest mean tumor-region entropy,
then render each as the same paper-style 5-panel figure used in
uncertainty_vis_v2_1/examples/ (T1c | T1c+GT | T1c+MC-Pred | Entropy | T1c+Entropy).

Two-phase to keep memory bounded:
  1. Screening pass over --n_screen val subjects: run MC inference, keep only each
     subject's mean entropy over the tumor-union mask (GT ∪ Pred), discard the rest.
  2. Re-run MC inference on just the highest- and lowest-entropy subject found, and
     save their figures.

Output: results_summary/examples_entropy_extremes/{high,low}_entropy_<subject>.png
"""

import argparse
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch

from model_v2 import UNet3DAttn, mc_inference
from visualize_uncertainty_v2 import get_val_files, make_subject_figure, make_entropy_histogram


def load_model(checkpoint, init_features, device):
    ckpt = torch.load(checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = UNet3DAttn(
        in_channels=4, out_channels=4,
        init_features=ckpt_args.get("init_features", init_features),
        dropout_p=ckpt_args.get("dropout", 0.15),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded: epoch {ckpt.get('epoch')}  best Dice {ckpt.get('best_dice'):.4f}")
    return model


def run_subject(model, fpath, device, mc_passes):
    with h5py.File(fpath, "r") as f:
        images_np = f["images"][:]
        gt_np     = f["seg"][:].astype(np.int64)

    images_t = torch.from_numpy(images_np).unsqueeze(0).to(device)
    mean_pred, entropy_t = mc_inference(model, images_t, n_passes=mc_passes)
    pred_class = mean_pred.argmax(dim=1)[0].cpu().numpy()
    entropy_np = entropy_t[0].cpu().numpy()

    def dice_bin(a, b):
        i = (a & b).sum(); d = a.sum() + b.sum()
        return 2 * i / d if d > 0 else 1.0
    dice = np.array([
        dice_bin((pred_class == 1) | (pred_class == 3), (gt_np == 1) | (gt_np == 3)),
        dice_bin( pred_class > 0,  gt_np > 0),
        dice_bin( pred_class == 3, gt_np == 3),
    ])
    return images_np, gt_np, pred_class, entropy_np, dice


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     default="checkpoints_v2_1/best_model.pth")
    p.add_argument("--data_dir",       default="/home/sahar/CV_medical_data_project/processed/train")
    p.add_argument("--val_split_json", default=None)
    p.add_argument("--init_features",  type=int, default=32)
    p.add_argument("--out_dir",        default="results_summary/examples_entropy_extremes")
    p.add_argument("--mc_passes",      type=int, default=20)
    p.add_argument("--n_screen",       type=int, default=50)
    p.add_argument("--screen_seed",    type=int, default=7)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, args.init_features, device)

    split_path = args.val_split_json or os.path.join(os.path.dirname(args.checkpoint), "val_split.json")
    val_files = get_val_files(args.data_dir, split_path)
    print(f"Val split: {len(val_files)} subjects ({split_path})")

    rng = random.Random(args.screen_seed)
    screen_files = rng.sample(val_files, min(args.n_screen, len(val_files)))
    print(f"\nScreening {len(screen_files)} subjects for entropy extremes ({args.mc_passes} MC passes each)...")

    scores = []
    for i, fpath in enumerate(screen_files, 1):
        images_np, gt_np, pred_class, entropy_np, dice = run_subject(model, fpath, device, args.mc_passes)
        tumor_mask = (gt_np > 0) | (pred_class > 0)
        mean_ent = float(entropy_np[tumor_mask].mean()) if tumor_mask.any() else 0.0
        scores.append((fpath, mean_ent))
        print(f"  [{i}/{len(screen_files)}] {Path(fpath).stem:35s} "
              f"tumor-mean-entropy={mean_ent:.5f}  Dice(TC/WT/ET)={dice[0]:.3f}/{dice[1]:.3f}/{dice[2]:.3f}")

    scores.sort(key=lambda t: t[1])
    low_path,  low_ent  = scores[0]
    high_path, high_ent = scores[-1]
    print(f"\nLowest  tumor-mean-entropy: {Path(low_path).stem}   ({low_ent:.5f})")
    print(f"Highest tumor-mean-entropy: {Path(high_path).stem}   ({high_ent:.5f})")

    print("\nRe-running MC inference on the two selected subjects for final figures...")
    picks = [("low_entropy", low_path), ("high_entropy", high_path)]
    subjects = {}
    for tag, fpath in picks:
        images_np, gt_np, pred_class, entropy_np, dice = run_subject(model, fpath, device, args.mc_passes)
        subjects[tag] = {
            "name": fpath, "images": images_np, "gt": gt_np,
            "pred": pred_class, "entropy": entropy_np, "dice": dice,
        }

    entropy_max_global = max(s["entropy"].max() for s in subjects.values())
    for tag, subj in subjects.items():
        subj_name = Path(subj["name"]).stem
        out_path = out_dir / f"{tag}_{subj_name}.png"
        make_subject_figure(subj, out_path, args.mc_passes, entropy_max_global)

        hist_path = out_dir / f"{tag}_{subj_name}_entropy_histogram.png"
        make_entropy_histogram(subj, hist_path, args.mc_passes, entropy_max_global)

        tumor_mask = (subj["gt"] > 0) | (subj["pred"] > 0)
        tumor_hist_path = out_dir / f"{tag}_{subj_name}_entropy_histogram_tumor_only.png"
        make_entropy_histogram(subj, tumor_hist_path, args.mc_passes, entropy_max_global,
                                region_mask=tumor_mask, region_label="tumor region only (GT ∪ Pred)")

    summary = {
        "checkpoint": args.checkpoint,
        "mc_passes": args.mc_passes,
        "n_screened": len(screen_files),
        "screen_seed": args.screen_seed,
        "low_entropy_subject":  {"name": Path(low_path).stem,  "tumor_mean_entropy": low_ent},
        "high_entropy_subject": {"name": Path(high_path).stem, "tumor_mean_entropy": high_ent},
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nDone. Output in {out_dir}/")


if __name__ == "__main__":
    main()