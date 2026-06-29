"""
inference_report.py
Two outputs for the held-out validation split:

  1. out_dir/slices/<subject>.png
       Every tumor-containing axial slice, 3 columns:
         GT overlay on T1c | Pred overlay on T1c | Diff map (TP/FP/FN, WT)

  2. out_dir/metrics_table.png
       BraTS-style aggregate table: mean ± std per region (TC / WT / ET)
       for Dice (DSC) and HD95, plus a per-subject breakdown.

Usage:
  python inference_report.py --checkpoint checkpoints/best_model.pth \\
                              --data_dir processed/train \\
                              [--n_subjects N]   # omit = all 324 val subjects
"""

import argparse
import os
import random
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from monai.metrics import HausdorffDistanceMetric

from model import UNet3D, mc_inference, get_region_masks


# ── Color maps ───────────────────────────────────────────────────────────────
SEG_CMAP = mcolors.ListedColormap(["black", "red", "yellow", "cyan"])
SEG_NORM = mcolors.BoundaryNorm([0, 1, 2, 3, 4], SEG_CMAP.N)
BG = "#1a1a2e"

REGION_NAMES  = ["TC", "WT", "ET"]
REGION_COLORS = ["#e05c5c", "#52aacc", "#52cc8a"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_val_files(data_dir, val_split=0.2, seed=42):
    files = sorted(glob(os.path.join(data_dir, "*.h5")))
    rng = random.Random(seed)
    rng.shuffle(files)
    return files[: int(len(files) * val_split)]


def normalize_slice(s):
    nz = s[s > 0]
    if nz.size == 0:
        return s
    p1, p99 = np.percentile(nz, [1, 99])
    return np.clip((s - p1) / (p99 - p1 + 1e-8), 0, 1)


def diff_map_rgb(gt_bin, pred_bin):
    """TP=green, FP=orange, FN=blue on black."""
    rgb = np.zeros((*gt_bin.shape, 3), dtype=np.float32)
    rgb[(gt_bin == 1) & (pred_bin == 1)] = [0.20, 0.85, 0.25]   # TP
    rgb[(gt_bin == 0) & (pred_bin == 1)] = [1.00, 0.45, 0.05]   # FP
    rgb[(gt_bin == 1) & (pred_bin == 0)] = [0.25, 0.45, 1.00]   # FN
    return rgb


def per_subject_dice(pred, gt):
    def dice(a, b):
        i = (a & b).sum(); d = a.sum() + b.sum()
        return 2 * i / d if d > 0 else 1.0
    return np.array([
        dice((pred == 1) | (pred == 3), (gt == 1) | (gt == 3)),  # TC
        dice(pred > 0, gt > 0),                                    # WT
        dice(pred == 3, gt == 3),                                  # ET
    ])


def per_subject_hd95(pred_t, gt_t):
    metric = HausdorffDistanceMetric(
        include_background=True, percentile=95, reduction="mean_batch"
    )
    TC_p, WT_p, ET_p = get_region_masks(pred_t.unsqueeze(0))
    TC_g, WT_g, ET_g = get_region_masks(gt_t.unsqueeze(0))
    metric(
        y_pred=torch.stack([TC_p, WT_p, ET_p], 1).float(),
        y=     torch.stack([TC_g, WT_g, ET_g], 1).float(),
    )
    scores = metric.aggregate().numpy()
    return np.where(np.isfinite(scores), scores, np.nan)


# ── Output 1: single representative axial slice per subject ──────────────────

def _best_axial_slice(gt_np, pred_np):
    """Return z index of the axial slice with the most combined tumor voxels."""
    combined = (gt_np > 0) | (pred_np > 0)
    counts   = combined.sum(axis=(0, 1))   # (Z,)
    return int(counts.argmax())


def save_subject_figure(images_np, gt_np, pred_np, out_path, subject_name, dice, hd95):
    """
    One figure per subject: the single most tumor-rich axial slice.

    Layout (1 row × 4 panels):
      T1c (MRI)  |  Ground Truth  |  Prediction  |  Difference map

    images_np : (4, X, Y, Z)   — channels: t1n, t1c, t2w, t2f
    gt_np     : (X, Y, Z)      — int labels {0, 1, 2, 3}
    pred_np   : (X, Y, Z)      — int labels {0, 1, 2, 3}
    """
    wt_gt   = (gt_np > 0).astype(np.uint8)
    wt_pred = (pred_np > 0).astype(np.uint8)

    z = _best_axial_slice(gt_np, pred_np)

    t1c_sl  = normalize_slice(images_np[1, :, :, z]).T   # (Y, X) — display oriented
    gt_sl   = gt_np[:, :, z].T.astype(np.float32)
    pred_sl = pred_np[:, :, z].T.astype(np.float32)
    diff_sl = diff_map_rgb(wt_gt[:, :, z].T, wt_pred[:, :, z].T)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5),
                             gridspec_kw={"wspace": 0.04})
    fig.patch.set_facecolor(BG)

    panels = [
        ("T1c (MRI)",      t1c_sl,   None,     None,     0.00),
        ("Ground Truth",   t1c_sl,   gt_sl,    None,     0.55),
        ("Prediction",     t1c_sl,   pred_sl,  None,     0.55),
        ("Difference Map", t1c_sl,   None,     diff_sl,  0.75),
    ]

    for ax, (title, mri, seg_overlay, diff_overlay, alpha) in zip(axes, panels):
        ax.imshow(mri, cmap="gray", origin="lower", interpolation="bilinear")
        if seg_overlay is not None:
            ax.imshow(seg_overlay, cmap=SEG_CMAP, norm=SEG_NORM,
                      origin="lower", alpha=alpha, interpolation="nearest")
        if diff_overlay is not None:
            ax.imshow(diff_overlay, origin="lower", alpha=alpha,
                      interpolation="nearest")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#3a3a5a")

    # Legend (bottom centre)
    legend_handles = [
        mpatches.Patch(color="red",     label="NCR — Necrotic Core (1)"),
        mpatches.Patch(color="yellow",  label="SNFH / Edema (2)"),
        mpatches.Patch(color="cyan",    label="ET — Enhancing Tumor (3)"),
        mpatches.Patch(color="#20d940", label="TP — correct"),
        mpatches.Patch(color="#ff7210", label="FP — over-segmented"),
        mpatches.Patch(color="#4070ff", label="FN — missed"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=6,
               fontsize=9, labelcolor="white", facecolor=BG, edgecolor="#555",
               framealpha=0.4, bbox_to_anchor=(0.5, -0.04))

    hd_str = "  ".join(
        f"{r} {v:.1f} mm" if np.isfinite(v) else f"{r} —"
        for r, v in zip(REGION_NAMES, hd95)
    )
    fig.suptitle(
        f"{subject_name}   ·   axial slice z = {z}\n"
        f"Dice   TC {dice[0]:.4f}   WT {dice[1]:.4f}   ET {dice[2]:.4f}   "
        f"Mean {float(np.nanmean(dice)):.4f}     |     HD95   {hd_str}",
        color="white", fontsize=10, fontweight="bold", y=1.04,
    )

    plt.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()


# ── Output 2: BraTS-style metrics table ──────────────────────────────────────

def save_metrics_table(all_dice, all_hd95, subject_names, out_path):
    """
    all_dice : (N, 3) — columns TC / WT / ET
    all_hd95 : (N, 3)
    Saves a compact BraTS-style table: one row per method, mean across all subjects.
    """
    N = len(subject_names)
    mean_d = np.nanmean(all_dice, axis=0)   # (3,) TC / WT / ET
    mean_h = np.nanmean(all_hd95, axis=0)

    col_headers = [
        "Method",
        "DSC TC ↑", "DSC WT ↑", "DSC ET ↑", "Mean DSC ↑",
        "HD95 TC ↓\n(mm)", "HD95 WT ↓\n(mm)", "HD95 ET ↓\n(mm)", "Mean HD95 ↓\n(mm)",
    ]

    rows = [[
        f"3D U-Net + MC Dropout   (n = {N})",
        f"{mean_d[0]:.4f}",
        f"{mean_d[1]:.4f}",
        f"{mean_d[2]:.4f}",
        f"{float(np.nanmean(mean_d)):.4f}",
        f"{mean_h[0]:.2f}",
        f"{mean_h[1]:.2f}",
        f"{mean_h[2]:.2f}",
        f"{float(np.nanmean(mean_h)):.2f}",
    ]]

    fig, ax = plt.subplots(figsize=(16, 3.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=col_headers,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 3.2)

    # Header row
    for j in range(len(col_headers)):
        c = tbl[0, j]
        c.set_facecolor("#0f3460")
        c.set_text_props(color="white", fontweight="bold")
        c.set_edgecolor("#556")

    # Data row
    for j in range(len(col_headers)):
        c = tbl[1, j]
        c.set_edgecolor("#445")
        if j == 0:
            c.set_facecolor("#131c2e")
            c.set_text_props(color="#ccddff", fontweight="bold")
        elif 1 <= j <= 3:       # DSC — green heat
            val = float(rows[0][j])
            intensity = min(max((val - 0.60) / 0.35, 0), 1)
            c.set_facecolor((0.04 + 0.08 * intensity,
                             0.15 + 0.40 * intensity,
                             0.04 + 0.08 * intensity))
            c.set_text_props(color="white", fontweight="bold")
        elif j == 4:            # Mean DSC
            c.set_facecolor("#1a3a5c")
            c.set_text_props(color="#ffcc44", fontweight="bold")
        elif 5 <= j <= 7:       # HD95 — inverse green heat (lower = greener)
            val = float(rows[0][j])
            intensity = min(max(1.0 - (val - 1.0) / 25.0, 0), 1)
            c.set_facecolor((0.04 + 0.08 * intensity,
                             0.15 + 0.35 * intensity,
                             0.04 + 0.08 * intensity))
            c.set_text_props(color="white", fontweight="bold")
        else:                   # Mean HD95
            c.set_facecolor("#1a3a5c")
            c.set_text_props(color="#ffcc44", fontweight="bold")

    ax.set_title(
        "BraTS 2024 GLI — Segmentation Results  |  "
        "Internal Validation — 20% holdout of training set (n=324), not used for gradient updates",
        color="white", fontsize=11, fontweight="bold", pad=16,
    )

    caption_lines = [
        "DSC (Dice Similarity Coefficient): segmentation overlap in [0, 1].  1 = perfect, 0 = no overlap.  Higher ↑ is better.",
        "HD95 (95th-percentile Hausdorff Distance, mm): worst-case surface distance after excluding the top 5% outlier points.  Lower ↓ is better.",
        "TC  (Tumor Core)   = Necrotic Core (NCR, label 1) + Enhancing Tumor (ET, label 3)",
        "WT  (Whole Tumor)  = NCR (1) + Surrounding FLAIR Hyperintensity / Edema (SNFH, label 2) + ET (3)",
        "ET  (Enhancing Tumor) = label 3 only    |    Mean = unweighted average of TC, WT, ET",
    ]
    fig.text(
        0.02, 0.01,
        "\n".join(caption_lines),
        color="#8899bb", fontsize=8.5, va="bottom", ha="left",
        fontfamily="monospace",
        transform=fig.transFigure,
    )

    plt.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Inference report: axial slices per subject + BraTS metrics table"
    )
    p.add_argument("--checkpoint",    default="checkpoints/best_model.pth")
    p.add_argument("--data_dir",      default="processed/train")
    p.add_argument("--val_split",     type=float, default=0.2)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--n_subjects",    type=int,   default=None,
                   help="Limit to first N subjects (default: all val subjects)")
    p.add_argument("--out_dir",       default="exploration_output/inference_report")
    p.add_argument("--init_features", type=int,   default=32)
    p.add_argument("--mc_passes",     type=int,   default=0,
                   help="MC Dropout passes (0 = deterministic)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir    = Path(args.out_dir)
    slices_dir = out_dir / "slices"
    out_dir.mkdir(parents=True, exist_ok=True)
    slices_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt      = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = UNet3D(
        in_channels=4, out_channels=4,
        init_features=ckpt_args.get("init_features", args.init_features),
        dropout_p=ckpt_args.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded: epoch {ckpt.get('epoch')}  best Dice {ckpt.get('best_dice'):.4f}")
    print(f"Inference mode: {'MC Dropout (' + str(args.mc_passes) + ' passes)' if args.mc_passes > 0 else 'deterministic'}")

    val_files = get_val_files(args.data_dir, args.val_split, args.seed)
    if args.n_subjects:
        val_files = val_files[: args.n_subjects]
    print(f"\nRunning on {len(val_files)} subjects → {out_dir}\n")

    all_dice   = []
    all_hd95   = []
    subj_names = []

    for i, fpath in enumerate(val_files, 1):
        name = Path(fpath).stem
        print(f"  [{i:03d}/{len(val_files)}] {name}", end="", flush=True)

        with h5py.File(fpath, "r") as f:
            images_np = f["images"][:]
            gt_np     = f["seg"][:].astype(np.int64)

        images_t = torch.from_numpy(images_np).unsqueeze(0).to(device)
        if args.mc_passes > 0:
            mean_pred, _ = mc_inference(model, images_t, n_passes=args.mc_passes)
            pred_t = mean_pred.argmax(dim=1)[0].cpu()
        else:
            with torch.no_grad():
                pred_t = model(images_t).argmax(dim=1)[0].cpu()

        pred_np = pred_t.numpy()
        gt_t    = torch.from_numpy(gt_np)

        dice = per_subject_dice(pred_np, gt_np)
        hd95 = per_subject_hd95(pred_t, gt_t)

        all_dice.append(dice)
        all_hd95.append(hd95)
        subj_names.append(fpath)

        out_path = slices_dir / f"{name}.png"
        save_subject_figure(images_np, gt_np, pred_np, out_path, name, dice, hd95)
        print(f"  Dice {np.nanmean(dice):.4f}  → {out_path.name}")

    all_dice = np.stack(all_dice)   # (N, 3)
    all_hd95 = np.stack(all_hd95)  # (N, 3)

    print("\nGenerating metrics table ...")
    save_metrics_table(all_dice, all_hd95, val_files, out_dir / "metrics_table.png")

    print(f"\nDone.")
    print(f"  {len(val_files)} axial-slice figures → {slices_dir}/")
    print(f"  Metrics table                       → {out_dir}/metrics_table.png")


if __name__ == "__main__":
    main()