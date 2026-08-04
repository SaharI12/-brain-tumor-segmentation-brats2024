"""
Same calibration analysis as visualize_uncertainty_v2.py (reliability diagram + ROC),
but restricted to tumor-relevant voxels only: TP + FP + FN (whole-tumor binary),
excluding TN (correctly-predicted non-tumor background/healthy tissue).

Why: the brain-mask-restricted version in visualize_uncertainty_v2.py still lets TN
voxels dominate the pooled sample (mean entropy ~0.0002, essentially always confident
and correct), which pulls ECE down regardless of how calibrated the model is on the
actually-hard voxels (tumor and near-boundary). This script answers "how calibrated
is the model where it might actually be wrong?"

Output: results_summary/calibration_foreground_only.png, results_summary/calibration_foreground_only.json
"""

import argparse
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from model_v2 import UNet3DAttn, mc_inference
from visualize_uncertainty_v2 import get_val_files, expected_calibration_error


def collect_calibration_points_foreground(model, val_files, device, n_subjects, mc_passes,
                                            subsample_per_subject, seed):
    """
    Same as collect_calibration_points in visualize_uncertainty_v2.py, but keeps only
    voxels where gt_bin | pred_bin (whole-tumor union) — i.e. TP + FP + FN, dropping TN.
    """
    rng = random.Random(seed)
    sampled = val_files if n_subjects >= len(val_files) else rng.sample(val_files, n_subjects)

    all_conf, all_correct, all_entropy = [], [], []

    for i, fpath in enumerate(sampled, 1):
        print(f"  [calib {i}/{len(sampled)}] {Path(fpath).name}")
        with h5py.File(fpath, "r") as f:
            images_np = f["images"][:]
            gt_np     = f["seg"][:].astype(np.int64)

        images_t = torch.from_numpy(images_np).unsqueeze(0).to(device)
        mean_pred, entropy_t = mc_inference(model, images_t, n_passes=mc_passes)

        conf_np    = mean_pred.max(dim=1).values[0].cpu().numpy()
        pred_class = mean_pred.argmax(dim=1)[0].cpu().numpy()
        entropy_np = entropy_t[0].cpu().numpy()
        correct_np = (pred_class == gt_np).astype(np.float32)

        gt_bin   = gt_np > 0
        pred_bin = pred_class > 0
        fg_mask = gt_bin | pred_bin  # TP + FP + FN, excludes TN

        idx = np.flatnonzero(fg_mask)
        if idx.size == 0:
            continue
        if idx.size > subsample_per_subject:
            idx = np.array(rng.sample(list(idx), subsample_per_subject))

        all_conf.append(conf_np.ravel()[idx])
        all_correct.append(correct_np.ravel()[idx])
        all_entropy.append(entropy_np.ravel()[idx])

    return (np.concatenate(all_conf), np.concatenate(all_correct),
            np.concatenate(all_entropy))


def make_calibration_plot(confidence, correct, entropy, out_path: Path, n_bins=15):
    BG = "black"
    ece, bin_conf, bin_acc, bin_count, bin_edges = expected_calibration_error(confidence, correct, n_bins=n_bins)

    is_error = 1.0 - correct
    auroc = roc_auc_score(is_error, entropy)
    fpr, tpr, _ = roc_curve(is_error, entropy)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    ax.set_facecolor(BG)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valid = ~np.isnan(bin_acc)
    ax.bar(bin_centers[valid], bin_acc[valid], width=bin_width * 0.9,
           color="#3dcc9a", edgecolor="none", alpha=0.85, label="Empirical accuracy")
    ax.plot([0, 1], [0, 1], "--", color="#ff7320", linewidth=1.5, label="Perfect calibration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (max softmax prob)", color="white", fontsize=10)
    ax.set_ylabel("Empirical accuracy", color="white", fontsize=10)
    ax.set_title(f"Reliability Diagram — Foreground Only  (ECE = {ece:.4f})", color="white",
                 fontsize=11, fontweight="bold")
    ax.tick_params(colors="white", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#555")
    ax.grid(color="#333", linewidth=0.7)
    ax.legend(fontsize=9, labelcolor="white", facecolor=BG, edgecolor="#555", loc="upper left")

    ax = axes[1]
    ax.set_facecolor(BG)
    ax.plot(fpr, tpr, color="#3dcc9a", linewidth=2, label=f"Entropy (AUROC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="#ff7320", linewidth=1.5, label="Chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate", color="white", fontsize=10)
    ax.set_ylabel("True positive rate", color="white", fontsize=10)
    ax.set_title("Entropy as an Error Detector — Foreground Only (ROC)", color="white",
                 fontsize=11, fontweight="bold")
    ax.tick_params(colors="white", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#555")
    ax.grid(color="#333", linewidth=0.7)
    ax.legend(fontsize=9, labelcolor="white", facecolor=BG, edgecolor="#555", loc="lower right")

    plt.tight_layout(pad=1.5)
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")

    bins_table = [
        {"conf_center": float(c), "accuracy": float(a), "count": int(n)}
        for c, a, n in zip(bin_centers, bin_acc, bin_count) if n > 0
    ]
    return ece, auroc, bins_table


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     default="checkpoints_v2_1/best_model.pth")
    p.add_argument("--data_dir",       default="/home/sahar/CV_medical_data_project/processed/train")
    p.add_argument("--val_split_json", default=None)
    p.add_argument("--init_features",  type=int, default=32)
    p.add_argument("--out_dir",        default="results_summary")
    p.add_argument("--mc_passes",      type=int, default=20)
    p.add_argument("--n_calib_subjects", type=int, default=50)
    p.add_argument("--calib_subsample_per_subject", type=int, default=5000)
    p.add_argument("--calib_seed",     type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt      = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = UNet3DAttn(
        in_channels=4, out_channels=4,
        init_features=ckpt_args.get("init_features", args.init_features),
        dropout_p=ckpt_args.get("dropout", 0.15),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded: epoch {ckpt.get('epoch')}  best Dice {ckpt.get('best_dice'):.4f}")

    split_path = args.val_split_json or os.path.join(os.path.dirname(args.checkpoint), "val_split.json")
    val_files = get_val_files(args.data_dir, split_path)

    print(f"\nRunning foreground-only calibration analysis over "
          f"{min(args.n_calib_subjects, len(val_files))} subjects...")
    confidence, correct, entropy = collect_calibration_points_foreground(
        model, val_files, device,
        n_subjects=args.n_calib_subjects, mc_passes=args.mc_passes,
        subsample_per_subject=args.calib_subsample_per_subject, seed=args.calib_seed,
    )
    print(f"Pooled voxels: {len(confidence)}")

    ece, auroc, bins_table = make_calibration_plot(
        confidence, correct, entropy, out_dir / "calibration_foreground_only.png")

    print(f"\nECE (foreground only):   {ece:.4f}")
    print(f"AUROC (foreground only): {auroc:.4f}")

    summary = {
        "checkpoint": args.checkpoint,
        "mc_passes": args.mc_passes,
        "n_calib_subjects": min(args.n_calib_subjects, len(val_files)),
        "n_voxels_pooled": int(len(confidence)),
        "mean_accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
        "ece": ece,
        "auroc_entropy_error_detector": auroc,
        "bins": bins_table,
    }
    with open(out_dir / "calibration_foreground_only.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_dir / 'calibration_foreground_only.json'}")


if __name__ == "__main__":
    main()
