"""
Evaluation visualization: one figure per subject + summary table.
Outputs to exploration_output/eval_vis/
  - subjects/subject_XX_<name>.png  : one per subject (3 axes × 4 panels + metrics)
  - eval_table.png                  : per-subject Dice + HD95 table + overall row
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
import matplotlib.gridspec as gridspec

from monai.metrics import HausdorffDistanceMetric

from model import UNet3D, get_region_masks


# ── Config ──────────────────────────────────────────────────────────────────────
OUT_DIR    = Path("exploration_output/eval_vis")
SUBJ_DIR   = OUT_DIR / "subjects"
N_SUBJECTS = 20
VIZ_SEED   = 13

SEG_CMAP = mcolors.ListedColormap(["black", "red", "yellow", "cyan"])
SEG_NORM = mcolors.BoundaryNorm([0, 1, 2, 3, 4], SEG_CMAP.N)

OVERALL_DICE = np.array([0.8638, 0.9087, 0.8503])
OVERALL_HD95 = np.array([5.65,   6.11,   5.74])
REGION_NAMES = ["TC", "WT", "ET"]
REGION_COLORS = ["#e05c5c", "#52aacc", "#52cc8a"]

BG = "#1a1a2e"


# ── Helpers ──────────────────────────────────────────────────────────────────────

def get_val_files(data_dir, val_split=0.2, seed=42):
    files = sorted(glob(os.path.join(data_dir, "*.h5")))
    rng = random.Random(seed)
    rng.shuffle(files)
    return files[: int(len(files) * val_split)]


def normalize_slice(s: np.ndarray) -> np.ndarray:
    nz = s[s > 0]
    if nz.size == 0:
        return s
    p1, p99 = np.percentile(nz, [1, 99])
    return np.clip((s - p1) / (p99 - p1 + 1e-8), 0, 1)


def get_centroid(seg: np.ndarray) -> tuple:
    coords = np.argwhere(seg > 0)
    return tuple(coords.mean(axis=0).astype(int)) if len(coords) else tuple(s // 2 for s in seg.shape)


def get_slices(vol3d: np.ndarray, cx: int, cy: int, cz: int) -> dict:
    """Return display-oriented 2-D slices through the centroid for each axis."""
    return {
        "Axial":    vol3d[:, :, cz].T,
        "Coronal":  vol3d[:, cy, :].T,
        "Sagittal": vol3d[cx, :, :].T,
    }


def error_map_rgb(gt_bin: np.ndarray, pred_bin: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*gt_bin.shape, 3), dtype=np.float32)
    rgb[(gt_bin == 1) & (pred_bin == 1)] = [0.20, 0.85, 0.25]  # TP green
    rgb[(gt_bin == 0) & (pred_bin == 1)] = [1.00, 0.45, 0.05]  # FP orange
    rgb[(gt_bin == 1) & (pred_bin == 0)] = [0.25, 0.45, 1.00]  # FN blue
    return rgb


def per_subject_dice(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    def dice(a, b):
        i = (a & b).sum(); d = a.sum() + b.sum()
        return 2 * i / d if d > 0 else 1.0
    return np.array([
        dice((pred == 1) | (pred == 3), (gt == 1) | (gt == 3)),
        dice( pred > 0,  gt > 0),
        dice( pred == 3, gt == 3),
    ])


def per_subject_hd95(pred_t: torch.Tensor, gt_t: torch.Tensor) -> np.ndarray:
    metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")
    TC_p, WT_p, ET_p = get_region_masks(pred_t.unsqueeze(0))
    TC_g, WT_g, ET_g = get_region_masks(gt_t.unsqueeze(0))
    metric(y_pred=torch.stack([TC_p, WT_p, ET_p], 1).float(),
           y=    torch.stack([TC_g, WT_g, ET_g], 1).float())
    scores = metric.aggregate().numpy()
    return np.where(np.isfinite(scores), scores, np.nan)


# ── Per-subject figure ────────────────────────────────────────────────────────────

def make_subject_figure(subj: dict, out_path: Path, subject_idx: int, total: int):
    """
    3 rows (Axial / Coronal / Sagittal) × 4 image panels + right sidebar with metrics.
    Panels: T1c | GT overlay | Pred overlay | Error map (WT)
    """
    imgs = subj["images"]   # (4, X, Y, Z)
    gt   = subj["gt"]       # (X, Y, Z)
    pred = subj["pred"]     # (X, Y, Z)
    dice = subj["dice"]
    hd95 = subj["hd95"]
    name = Path(subj["name"]).stem

    cx, cy, cz = get_centroid(gt)

    axes_order  = ["Axial", "Coronal", "Sagittal"]
    col_titles  = ["T1c (MRI)", "Ground Truth", "Prediction", "Error Map (WT)"]
    n_img_cols  = 4

    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor(BG)

    # Main grid: 3 image rows + 1 header row | image cols + sidebar
    gs = gridspec.GridSpec(
        3, n_img_cols + 1,
        figure=fig,
        hspace=0.06, wspace=0.04,
        width_ratios=[1, 1, 1, 1, 0.85],
    )

    wt_gt   = gt > 0
    wt_pred = pred > 0

    for row_i, axis_name in enumerate(axes_order):
        t1c_sl  = normalize_slice(get_slices(imgs[1], cx, cy, cz)[axis_name])
        gt_sl   = get_slices(gt.astype(np.float32),   cx, cy, cz)[axis_name]
        pred_sl = get_slices(pred.astype(np.float32),  cx, cy, cz)[axis_name]
        err_sl  = error_map_rgb(
            get_slices(wt_gt.astype(np.uint8),   cx, cy, cz)[axis_name],
            get_slices(wt_pred.astype(np.uint8), cx, cy, cz)[axis_name],
        )

        panels_row = []

        # T1c
        ax = fig.add_subplot(gs[row_i, 0])
        ax.imshow(t1c_sl, cmap="gray", origin="lower")
        panels_row.append(ax)

        # GT overlay
        ax = fig.add_subplot(gs[row_i, 1])
        ax.imshow(t1c_sl, cmap="gray", origin="lower")
        ax.imshow(gt_sl, cmap=SEG_CMAP, norm=SEG_NORM, origin="lower",
                  alpha=0.55, interpolation="nearest")
        panels_row.append(ax)

        # Pred overlay
        ax = fig.add_subplot(gs[row_i, 2])
        ax.imshow(t1c_sl, cmap="gray", origin="lower")
        ax.imshow(pred_sl, cmap=SEG_CMAP, norm=SEG_NORM, origin="lower",
                  alpha=0.55, interpolation="nearest")
        panels_row.append(ax)

        # Error map
        ax = fig.add_subplot(gs[row_i, 3])
        ax.imshow(t1c_sl, cmap="gray", origin="lower")
        ax.imshow(err_sl, origin="lower", alpha=0.70)
        panels_row.append(ax)

        for col_i, ax in enumerate(panels_row):
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#444")
            if row_i == 0:
                ax.set_title(col_titles[col_i], color="white", fontsize=10,
                             fontweight="bold", pad=5)
            if col_i == 0:
                ax.set_ylabel(axis_name, color="white", fontsize=10,
                              fontweight="bold", labelpad=6)

    # ── Sidebar: metrics ────────────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[:, 4])   # spans all 3 rows
    ax_bar.set_facecolor(BG)

    # Dice bars
    y_pos = np.array([0.78, 0.54, 0.30])
    bar_h = 0.16
    for i, (region, color, d_val) in enumerate(zip(REGION_NAMES, REGION_COLORS, dice)):
        ax_bar.barh(y_pos[i], d_val, height=bar_h, color=color, edgecolor="none",
                    left=0, zorder=3)
        ax_bar.barh(y_pos[i], 1.0,   height=bar_h, color="#2a2a3e", edgecolor="none",
                    left=0, zorder=2)
        ax_bar.text(0.02, y_pos[i], region,
                    va="center", ha="left", fontsize=10, color="white", fontweight="bold", zorder=4)
        ax_bar.text(min(d_val + 0.03, 0.96), y_pos[i], f"{d_val:.4f}",
                    va="center", ha="left", fontsize=9, color="white", zorder=4)

    # Overall mean reference line
    ax_bar.axvline(OVERALL_DICE.mean(), color="#ffcc44", lw=1.4, linestyle="--", alpha=0.8, zorder=5)
    ax_bar.text(OVERALL_DICE.mean() + 0.01, 0.92, f"Overall\n{OVERALL_DICE.mean():.3f}",
                color="#ffcc44", fontsize=7.5, va="top")

    ax_bar.set_xlim(0, 1.05)
    ax_bar.set_ylim(0.10, 1.00)
    ax_bar.set_title("Dice Score", color="white", fontsize=10, fontweight="bold", pad=5)

    # HD95 text block
    hd_lines = [f"HD95 (mm)"]
    for region, h_val in zip(REGION_NAMES, hd95):
        hd_lines.append(f"  {region}: {h_val:.2f}" if np.isfinite(h_val) else f"  {region}: —")
    hd_lines.append(f"  Mean: {np.nanmean(hd95):.2f}")
    ax_bar.text(0.02, 0.18, "\n".join(hd_lines),
                transform=ax_bar.transAxes,
                va="bottom", ha="left", fontsize=9, color="#aaccff",
                fontfamily="monospace")

    ax_bar.set_xticks([]); ax_bar.set_yticks([])
    for sp in ax_bar.spines.values():
        sp.set_color("#333")

    # ── Legend ───────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="red",     label="NCR (1)"),
        mpatches.Patch(color="yellow",  label="SNFH/Edema (2)"),
        mpatches.Patch(color="cyan",    label="ET (3)"),
        mpatches.Patch(color="#20d940", label="TP"),
        mpatches.Patch(color="#ff7210", label="FP (over-seg)"),
        mpatches.Patch(color="#4070ff", label="FN (missed)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=6,
               fontsize=8.5, labelcolor="white", facecolor=BG,
               edgecolor="#555", framealpha=0.3,
               bbox_to_anchor=(0.46, -0.02))

    mean_dice = float(np.nanmean(dice))
    fig.suptitle(
        f"[{subject_idx:02d}/{total}]  {name}  |  "
        f"Mean Dice: {mean_dice:.4f}   "
        f"TC {dice[0]:.4f}  WT {dice[1]:.4f}  ET {dice[2]:.4f}",
        color="white", fontsize=11, fontweight="bold", y=1.01,
    )

    plt.savefig(str(out_path), dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ── Summary Table ────────────────────────────────────────────────────────────────

def make_table(rows: list, out_path: Path):
    col_headers = [
        "Subject", "TC Dice", "WT Dice", "ET Dice", "Mean Dice",
        "TC HD95", "WT HD95", "ET HD95", "Mean HD95",
    ]
    table_data = []
    for r in rows:
        d, h = r["dice"], r["hd95"]
        table_data.append([
            Path(r["name"]).stem,
            f"{d[0]:.4f}", f"{d[1]:.4f}", f"{d[2]:.4f}", f"{float(np.nanmean(d)):.4f}",
            f"{h[0]:.2f}" if np.isfinite(h[0]) else "—",
            f"{h[1]:.2f}" if np.isfinite(h[1]) else "—",
            f"{h[2]:.2f}" if np.isfinite(h[2]) else "—",
            f"{float(np.nanmean(h)):.2f}" if np.any(np.isfinite(h)) else "—",
        ])

    table_data.append([
        "OVERALL (n=324)",
        f"{OVERALL_DICE[0]:.4f}", f"{OVERALL_DICE[1]:.4f}", f"{OVERALL_DICE[2]:.4f}",
        f"{OVERALL_DICE.mean():.4f}",
        f"{OVERALL_HD95[0]:.2f}", f"{OVERALL_HD95[1]:.2f}", f"{OVERALL_HD95[2]:.2f}",
        f"{OVERALL_HD95.mean():.2f}",
    ])

    n_rows = len(table_data)
    fig, ax = plt.subplots(figsize=(20, 0.52 * n_rows + 2.0))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    tbl = ax.table(cellText=table_data, colLabels=col_headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    for col_j in range(len(col_headers)):
        cell = tbl[0, col_j]
        cell.set_facecolor("#16213e")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#555")

    for row_i in range(1, n_rows + 1):
        is_overall = row_i == n_rows
        for col_j in range(len(col_headers)):
            cell = tbl[row_i, col_j]
            cell.set_edgecolor("#444")
            if is_overall:
                cell.set_facecolor("#0f3460")
                cell.set_text_props(color="#ffcc44", fontweight="bold")
            else:
                cell.set_facecolor("#1e2a3a" if row_i % 2 == 0 else "#16213e")
                if 1 <= col_j <= 3:
                    try:
                        val = float(table_data[row_i - 1][col_j])
                        intensity = min(max((val - 0.70) / 0.25, 0), 1)
                        cell.set_facecolor((0.05 + 0.15 * intensity,
                                            0.20 + 0.45 * intensity,
                                            0.05 + 0.15 * intensity))
                    except ValueError:
                        pass
                cell.set_text_props(color="white")

    ax.set_title(
        f"BraTS 2024 GLI — Segmentation Results  |  Epoch 113  |  Val split 20% (n=324)  |  Showing {n_rows - 1} sampled subjects",
        color="white", fontsize=11, fontweight="bold", pad=14,
    )
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default="checkpoints/best_model.pth")
    p.add_argument("--data_dir",      default="processed/train")
    p.add_argument("--val_split",     type=float, default=0.2)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--n_subjects",    type=int,   default=N_SUBJECTS)
    p.add_argument("--init_features", type=int,   default=32)
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBJ_DIR.mkdir(parents=True, exist_ok=True)

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

    val_files = get_val_files(args.data_dir, args.val_split, args.seed)
    rng       = random.Random(VIZ_SEED)
    sampled   = rng.sample(val_files, min(args.n_subjects, len(val_files)))
    print(f"\nSampled {len(sampled)} subjects\n")

    subjects_data = []
    for i, fpath in enumerate(sampled, 1):
        name = Path(fpath).stem
        print(f"  [{i:02d}/{len(sampled)}] {name}")

        with h5py.File(fpath, "r") as f:
            images_np = f["images"][:]
            gt_np     = f["seg"][:].astype(np.int64)

        images_t = torch.from_numpy(images_np).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_t = model(images_t).argmax(dim=1)[0].cpu()

        gt_t    = torch.from_numpy(gt_np)
        pred_np = pred_t.numpy()
        dice    = per_subject_dice(pred_np, gt_np)
        hd95    = per_subject_hd95(pred_t, gt_t)

        subj = {"name": fpath, "images": images_np, "gt": gt_np,
                "pred": pred_np, "dice": dice, "hd95": hd95}
        subjects_data.append(subj)

        out_path = SUBJ_DIR / f"subject_{i:02d}_{name}.png"
        make_subject_figure(subj, out_path, i, len(sampled))
        print(f"           → {out_path.name}")

    print("\nGenerating summary table...")
    make_table(subjects_data, OUT_DIR / "eval_table.png")

    print(f"\nDone. {len(sampled)} figures + table in {OUT_DIR}/")


if __name__ == "__main__":
    main()