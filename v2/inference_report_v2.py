"""
inference_report_v2.py — v2.1 Attention U-Net version of inference_report.py

Two outputs for the held-out validation split (persistent val_split.json, n=324):

  1. out_dir/slices/<subject>.png
       Every subject's most tumor-rich axial slice, 4 columns:
         T1c | GT overlay | Pred overlay | Diff map (TP/FP/FN, WT)

  2. out_dir/metrics_table.png
       BraTS-style aggregate table: mean over all subjects for Dice (DSC)
       and HD95, TC / WT / ET / Mean.

MC Dropout mode (--mc_passes N, N > 0) additionally produces, when
--per_pass_table is set:

  3. out_dir/mc_per_pass_table.png  +  out_dir/mc_per_pass_table.csv
       One row per individual dropout pass (not the pass-averaged
       prediction): Dice + HD95 aggregated over all 324 subjects using
       ONLY that pass's own prediction. Final row = mean ± std over the
       N passes — this is the pass-to-pass variability induced by
       dropout stochasticity, distinct from the single MC mean-prediction
       result reported in metrics_table.png.

Usage:
  python inference_report_v2.py --checkpoint checkpoints_v2_1/best_model.pth \\
                                 --out_dir exploration_output/inference_report_v2_1
  python inference_report_v2.py --checkpoint checkpoints_v2_1/best_model.pth \\
                                 --mc_passes 20 --per_pass_table \\
                                 --out_dir exploration_output/montecarlo_v2_1
"""

import argparse
import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from monai.metrics import DiceMetric, HausdorffDistanceMetric

from model_v2 import UNet3DAttn, get_region_masks


# ── Color maps ───────────────────────────────────────────────────────────────
SEG_CMAP = mcolors.ListedColormap(["black", "red", "yellow", "cyan"])
SEG_NORM = mcolors.BoundaryNorm([0, 1, 2, 3, 4], SEG_CMAP.N)
BG = "#1a1a2e"

REGION_NAMES  = ["TC", "WT", "ET"]
REGION_COLORS = ["#e05c5c", "#52aacc", "#52cc8a"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_val_files(data_dir, split_path):
    with open(split_path) as f:
        saved = json.load(f)
    return [os.path.join(data_dir, fn) for fn in saved["val"]]


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
    combined = (gt_np > 0) | (pred_np > 0)
    counts   = combined.sum(axis=(0, 1))
    return int(counts.argmax())


def save_subject_figure(images_np, gt_np, pred_np, out_path, subject_name, dice, hd95):
    wt_gt   = (gt_np > 0).astype(np.uint8)
    wt_pred = (pred_np > 0).astype(np.uint8)

    z = _best_axial_slice(gt_np, pred_np)

    t1c_sl  = normalize_slice(images_np[1, :, :, z]).T
    gt_sl   = gt_np[:, :, z].T.astype(np.float32)
    pred_sl = pred_np[:, :, z].T.astype(np.float32)
    diff_sl = diff_map_rgb(wt_gt[:, :, z].T, wt_pred[:, :, z].T)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), gridspec_kw={"wspace": 0.04})
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


# ── Output 2: BraTS-style aggregate metrics table ─────────────────────────────

def save_metrics_table(all_dice, all_hd95, method_label, out_path):
    N = len(all_dice)
    mean_d = np.nanmean(all_dice, axis=0)
    mean_h = np.nanmean(all_hd95, axis=0)

    col_headers = [
        "Method",
        "DSC TC ↑", "DSC WT ↑", "DSC ET ↑", "Mean DSC ↑",
        "HD95 TC ↓\n(mm)", "HD95 WT ↓\n(mm)", "HD95 ET ↓\n(mm)", "Mean HD95 ↓\n(mm)",
    ]

    rows = [[
        f"{method_label}   (n = {N})",
        f"{mean_d[0]:.4f}", f"{mean_d[1]:.4f}", f"{mean_d[2]:.4f}",
        f"{float(np.nanmean(mean_d)):.4f}",
        f"{mean_h[0]:.2f}", f"{mean_h[1]:.2f}", f"{mean_h[2]:.2f}",
        f"{float(np.nanmean(mean_h)):.2f}",
    ]]

    fig, ax = plt.subplots(figsize=(16, 3.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=col_headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 3.2)

    for j in range(len(col_headers)):
        c = tbl[0, j]
        c.set_facecolor("#0f3460")
        c.set_text_props(color="white", fontweight="bold")
        c.set_edgecolor("#556")

    for j in range(len(col_headers)):
        c = tbl[1, j]
        c.set_edgecolor("#445")
        if j == 0:
            c.set_facecolor("#131c2e")
            c.set_text_props(color="#ccddff", fontweight="bold")
        elif 1 <= j <= 3:
            val = float(rows[0][j])
            intensity = min(max((val - 0.60) / 0.35, 0), 1)
            c.set_facecolor((0.04 + 0.08 * intensity, 0.15 + 0.40 * intensity, 0.04 + 0.08 * intensity))
            c.set_text_props(color="white", fontweight="bold")
        elif j == 4:
            c.set_facecolor("#1a3a5c")
            c.set_text_props(color="#ffcc44", fontweight="bold")
        elif 5 <= j <= 7:
            val = float(rows[0][j])
            intensity = min(max(1.0 - (val - 1.0) / 25.0, 0), 1)
            c.set_facecolor((0.04 + 0.08 * intensity, 0.15 + 0.35 * intensity, 0.04 + 0.08 * intensity))
            c.set_text_props(color="white", fontweight="bold")
        else:
            c.set_facecolor("#1a3a5c")
            c.set_text_props(color="#ffcc44", fontweight="bold")

    ax.set_title(
        "BraTS 2024 GLI — Segmentation Results (v2.1 Attention U-Net)  |  "
        "Internal Validation — persistent 20% holdout (n=324)",
        color="white", fontsize=11, fontweight="bold", pad=16,
    )

    caption_lines = [
        "DSC (Dice Similarity Coefficient): segmentation overlap in [0, 1].  1 = perfect, 0 = no overlap.  Higher ↑ is better.",
        "HD95 (95th-percentile Hausdorff Distance, mm): worst-case surface distance after excluding the top 5% outlier points.  Lower ↓ is better.",
        "TC  (Tumor Core)   = Necrotic Core (NCR, label 1) + Enhancing Tumor (ET, label 3)",
        "WT  (Whole Tumor)  = NCR (1) + Surrounding FLAIR Hyperintensity / Edema (SNFH, label 2) + ET (3)",
        "ET  (Enhancing Tumor) = label 3 only    |    Mean = unweighted average of TC, WT, ET",
    ]
    fig.text(0.02, 0.01, "\n".join(caption_lines), color="#8899bb", fontsize=8.5,
              va="bottom", ha="left", fontfamily="monospace", transform=fig.transFigure)

    plt.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Output 3 (MC only): per-pass table ────────────────────────────────────────

def save_per_pass_table(per_pass_dice, per_pass_hd95, out_path_png, out_path_csv):
    """
    per_pass_dice : (N_passes, 3) — TC / WT / ET, each already averaged over all subjects
    per_pass_hd95 : (N_passes, 3)
    """
    n_passes = per_pass_dice.shape[0]

    mean_dice = per_pass_dice.mean(axis=0); std_dice = per_pass_dice.std(axis=0)
    mean_hd   = np.nanmean(per_pass_hd95, axis=0); std_hd = np.nanstd(per_pass_hd95, axis=0)

    col_headers = ["Pass", "Dice TC", "Dice WT", "Dice ET", "Dice Mean",
                   "HD95 TC", "HD95 WT", "HD95 ET", "HD95 Mean"]

    rows = []
    for p in range(n_passes):
        d = per_pass_dice[p]; h = per_pass_hd95[p]
        rows.append([
            f"{p + 1}",
            f"{d[0]:.4f}", f"{d[1]:.4f}", f"{d[2]:.4f}", f"{float(d.mean()):.4f}",
            f"{h[0]:.2f}", f"{h[1]:.2f}", f"{h[2]:.2f}", f"{float(np.nanmean(h)):.2f}",
        ])
    rows.append([
        "Mean ± Std",
        f"{mean_dice[0]:.4f}±{std_dice[0]:.4f}",
        f"{mean_dice[1]:.4f}±{std_dice[1]:.4f}",
        f"{mean_dice[2]:.4f}±{std_dice[2]:.4f}",
        f"{float(mean_dice.mean()):.4f}±{float(std_dice.mean()):.4f}",
        f"{mean_hd[0]:.2f}±{std_hd[0]:.2f}",
        f"{mean_hd[1]:.2f}±{std_hd[1]:.2f}",
        f"{mean_hd[2]:.2f}±{std_hd[2]:.2f}",
        f"{float(np.nanmean(mean_hd)):.2f}±{float(np.nanmean(std_hd)):.2f}",
    ])

    # CSV
    with open(out_path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(col_headers)
        w.writerows(rows)

    # Styled PNG table
    fig, ax = plt.subplots(figsize=(14, 0.42 * (n_passes + 2) + 1))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    tbl = ax.table(cellText=rows, colLabels=col_headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.35)

    for j in range(len(col_headers)):
        c = tbl[0, j]
        c.set_facecolor("#0f3460")
        c.set_text_props(color="white", fontweight="bold")
        c.set_edgecolor("#556")

    for i in range(1, len(rows) + 1):
        is_summary = (i == len(rows))
        for j in range(len(col_headers)):
            c = tbl[i, j]
            c.set_edgecolor("#333")
            if is_summary:
                c.set_facecolor("#1a3a5c")
                c.set_text_props(color="#ffcc44", fontweight="bold")
            else:
                c.set_facecolor("#131c2e" if i % 2 == 0 else "#0d1420")
                c.set_text_props(color="#dde6ff")

    ax.set_title(
        f"MC Dropout — Per-Pass Dice / HD95 over {n_passes} passes  "
        "(each row = one stochastic forward pass, aggregated over n=324 val subjects)",
        color="white", fontsize=11, fontweight="bold", pad=14,
    )

    plt.savefig(str(out_path_png), dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Saved: {out_path_png}")
    print(f"  Saved: {out_path_csv}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="v2.1 inference report: axial slices per subject + BraTS metrics table"
    )
    p.add_argument("--checkpoint",     default="checkpoints_v2_1/best_model.pth")
    p.add_argument("--data_dir",       default="/home/sahar/CV_medical_data_project/processed/train")
    p.add_argument("--val_split_json", default=None,
                   help="Path to val_split.json (default: alongside checkpoint)")
    p.add_argument("--n_subjects",     type=int, default=None,
                   help="Limit to first N subjects (default: all val subjects)")
    p.add_argument("--out_dir",        required=True)
    p.add_argument("--init_features",  type=int, default=32)
    p.add_argument("--mc_passes",      type=int, default=0,
                   help="MC Dropout passes (0 = deterministic)")
    p.add_argument("--per_pass_table", action="store_true",
                   help="Also track per-pass Dice/HD95 (only meaningful with --mc_passes > 0)")
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
    model = UNet3DAttn(
        in_channels=4, out_channels=4,
        init_features=ckpt_args.get("init_features", args.init_features),
        dropout_p=ckpt_args.get("dropout", 0.15),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded: epoch {ckpt.get('epoch')}  best Dice {ckpt.get('best_dice'):.4f}")
    mc_on = args.mc_passes > 0
    print(f"Inference mode: {'MC Dropout (' + str(args.mc_passes) + ' passes)' if mc_on else 'deterministic'}")

    split_path = args.val_split_json or os.path.join(os.path.dirname(args.checkpoint), "val_split.json")
    val_files = get_val_files(args.data_dir, split_path)
    if args.n_subjects:
        val_files = val_files[: args.n_subjects]
    print(f"\nRunning on {len(val_files)} subjects (split: {split_path}) → {out_dir}\n")

    all_dice   = []
    all_hd95   = []

    track_per_pass = mc_on and args.per_pass_table
    if track_per_pass:
        pass_dice_metrics = [DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=True)
                              for _ in range(args.mc_passes)]
        pass_hd_metrics   = [HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")
                              for _ in range(args.mc_passes)]

    for i, fpath in enumerate(val_files, 1):
        name = Path(fpath).stem
        print(f"  [{i:03d}/{len(val_files)}] {name}", end="", flush=True)

        with h5py.File(fpath, "r") as f:
            images_np = f["images"][:]
            gt_np     = f["seg"][:].astype(np.int64)

        images_t = torch.from_numpy(images_np).unsqueeze(0).to(device)
        gt_t     = torch.from_numpy(gt_np)

        if mc_on:
            prior_training = model.training
            model.train()
            prob_sum = None
            if track_per_pass:
                TC_g, WT_g, ET_g = get_region_masks(gt_t.unsqueeze(0))
                gt_r_cpu = torch.stack([TC_g, WT_g, ET_g], dim=1).float()
            with torch.no_grad():
                for p in range(args.mc_passes):
                    probs = torch.softmax(model(images_t), dim=1)
                    prob_sum = probs if prob_sum is None else prob_sum + probs

                    if track_per_pass:
                        # Metrics only need small integer masks — do this on CPU so the
                        # per-pass float region-mask tensors don't add GPU memory pressure
                        # on top of the running mean-prediction accumulator (caused an
                        # OOM around subject 68/324 when kept on GPU).
                        pass_class_cpu = probs.argmax(dim=1).cpu()
                        TC_p, WT_p, ET_p = get_region_masks(pass_class_cpu)
                        pred_r_cpu = torch.stack([TC_p, WT_p, ET_p], dim=1).float()
                        pass_dice_metrics[p](y_pred=pred_r_cpu, y=gt_r_cpu)
                        pass_hd_metrics[p](y_pred=pred_r_cpu, y=gt_r_cpu)
                    del probs
            mean_pred = prob_sum / args.mc_passes
            pred_t = mean_pred.argmax(dim=1)[0].cpu()
            model.train(prior_training)
            del prob_sum, mean_pred
            torch.cuda.empty_cache()
        else:
            with torch.no_grad():
                pred_t = model(images_t).argmax(dim=1)[0].cpu()

        pred_np = pred_t.numpy()

        dice = per_subject_dice(pred_np, gt_np)
        hd95 = per_subject_hd95(pred_t, gt_t)

        all_dice.append(dice)
        all_hd95.append(hd95)

        out_path = slices_dir / f"{name}.png"
        save_subject_figure(images_np, gt_np, pred_np, out_path, name, dice, hd95)
        print(f"  Dice {np.nanmean(dice):.4f}  → {out_path.name}")

    all_dice = np.stack(all_dice)
    all_hd95 = np.stack(all_hd95)

    print("\nGenerating metrics table ...")
    method_label = f"Attention U-Net v2.1 {'+ MC Dropout (' + str(args.mc_passes) + ' passes)' if mc_on else '(deterministic)'}"
    save_metrics_table(all_dice, all_hd95, method_label, out_dir / "metrics_table.png")

    if track_per_pass:
        print("\nAggregating per-pass metrics ...")
        per_pass_dice = np.stack([m.aggregate().cpu().numpy() for m in pass_dice_metrics])   # (N_passes, 3)
        per_pass_hd_raw = [m.aggregate().cpu().numpy() for m in pass_hd_metrics]
        per_pass_hd   = np.stack([np.where(np.isfinite(a), a, np.nan) for a in per_pass_hd_raw])
        save_per_pass_table(per_pass_dice, per_pass_hd,
                             out_dir / "mc_per_pass_table.png",
                             out_dir / "mc_per_pass_table.csv")

    print(f"\nDone.")
    print(f"  {len(val_files)} axial-slice figures → {slices_dir}/")
    print(f"  Metrics table                       → {out_dir}/metrics_table.png")


if __name__ == "__main__":
    main()