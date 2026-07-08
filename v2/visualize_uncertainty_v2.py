"""
Uncertainty visualization for v2.1 (Attention U-Net): MC Dropout entropy maps
overlaid on MRI, plus a voxel-level calibration analysis.

Outputs to <out_dir> (default v2/uncertainty_vis_v2_1/):
  - examples/example_NN_<subject>.png : one figure per sampled subject, 5 panels each
                                        (axial at tumor centroid)
  - uncertainty_stats.png             : entropy distribution split by TP / FP / FN / TN
  - calibration.png                   : reliability diagram (ECE) + ROC curve (entropy as error detector)
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
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import ColorbarBase
from sklearn.metrics import roc_auc_score, roc_curve

from model_v2 import UNet3DAttn, mc_inference, get_region_masks


SEG_CMAP = mcolors.ListedColormap(["black", "red", "yellow", "cyan"])
SEG_NORM = mcolors.BoundaryNorm([0, 1, 2, 3, 4], SEG_CMAP.N)


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_val_files(data_dir, split_path):
    with open(split_path) as f:
        saved = json.load(f)
    return [os.path.join(data_dir, fn) for fn in saved["val"]]


def normalize_slice(s: np.ndarray) -> np.ndarray:
    nz = s[s > 0]
    if nz.size == 0:
        return s
    p1, p99 = np.percentile(nz, [1, 99])
    return np.clip((s - p1) / (p99 - p1 + 1e-8), 0, 1)


def get_centroid(seg: np.ndarray) -> tuple:
    coords = np.argwhere(seg > 0)
    return tuple(coords.mean(axis=0).astype(int)) if len(coords) else tuple(s // 2 for s in seg.shape)


def axial(vol, z):
    return vol[:, :, z].T


def entropy_stats(entropy_np: np.ndarray, gt: np.ndarray, pred: np.ndarray):
    """Return per-region entropy arrays: TN, TP, FP, FN (whole-tumor binary)."""
    gt_bin   = (gt > 0).astype(bool)
    pred_bin = (pred > 0).astype(bool)
    masks = {
        "TN": (~gt_bin) & (~pred_bin),
        "TP": ( gt_bin) & ( pred_bin),
        "FP": (~gt_bin) & ( pred_bin),
        "FN": ( gt_bin) & (~pred_bin),
    }
    stats = {}
    for name, mask in masks.items():
        vals = entropy_np[mask]
        stats[name] = vals if vals.size > 0 else np.array([0.0])
    return stats


# ── Example figures (one per subject) ────────────────────────────────────────

def make_subject_figure(subj: dict, out_path: Path, mc_passes: int, entropy_max_global: float):
    """
    Paper-style single-subject figure, 5 panels (axial, tumor centroid):
      T1c  |  T1c + GT  |  T1c + MC Pred  |  Entropy map  |  T1c + Entropy overlay
    """
    BG = "black"
    col_titles = ["T1c (MRI)", "T1c + GT", "T1c + MC Pred", f"Entropy ({mc_passes} passes)", "T1c + Entropy"]

    imgs, gt, pred, entropy, dice = subj["images"], subj["gt"], subj["pred"], subj["entropy"], subj["dice"]
    name = Path(subj["name"]).stem

    cx, cy, cz = get_centroid(gt)
    t1c_sl  = normalize_slice(axial(imgs[1], cz))
    gt_sl   = axial(gt.astype(np.float32), cz)
    pred_sl = axial(pred.astype(np.float32), cz)
    ent_sl  = axial(entropy, cz)

    fig = plt.figure(figsize=(20, 4.2))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(1, 5, figure=fig, wspace=0.04)

    panels = []

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(t1c_sl, cmap="gray", origin="lower")
    panels.append(ax)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(t1c_sl, cmap="gray", origin="lower")
    ax.imshow(gt_sl, cmap=SEG_CMAP, norm=SEG_NORM, origin="lower",
              alpha=0.55, interpolation="nearest")
    panels.append(ax)

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(t1c_sl, cmap="gray", origin="lower")
    ax.imshow(pred_sl, cmap=SEG_CMAP, norm=SEG_NORM, origin="lower",
              alpha=0.55, interpolation="nearest")
    panels.append(ax)

    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(np.zeros_like(t1c_sl), cmap="gray", origin="lower")
    ax.imshow(ent_sl, cmap="inferno", origin="lower",
              vmin=0, vmax=entropy_max_global, interpolation="bilinear")
    panels.append(ax)

    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(t1c_sl, cmap="gray", origin="lower")
    tumor_mask_sl = axial(((gt > 0) | (pred > 0)).astype(np.float32), cz)
    ent_masked = ent_sl * (tumor_mask_sl > 0)
    ax.imshow(ent_masked, cmap="inferno", origin="lower",
              vmin=0, vmax=entropy_max_global, alpha=0.75, interpolation="bilinear")
    panels.append(ax)

    for col_j, ax in enumerate(panels):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.set_title(col_titles[col_j], color="white", fontsize=10, fontweight="bold", pad=3)

    cbar_ax = fig.add_axes([0.915, 0.14, 0.012, 0.68])
    norm = mcolors.Normalize(vmin=0, vmax=entropy_max_global)
    cb = ColorbarBase(cbar_ax, cmap=plt.cm.inferno, norm=norm, orientation="vertical")
    cb.set_label("Predictive Entropy", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    cb.outline.set_edgecolor("#555")

    legend_entries = [
        mpatches.Patch(color="red",    label="NCR (1)"),
        mpatches.Patch(color="yellow", label="SNFH/Edema (2)"),
        mpatches.Patch(color="cyan",   label="ET (3)"),
    ]
    fig.legend(handles=legend_entries, loc="lower center", ncol=3,
               fontsize=8.5, labelcolor="white", facecolor=BG,
               edgecolor="#555", framealpha=0.3,
               bbox_to_anchor=(0.44, 0.01))

    fig.suptitle(
        f"{name}  —  Dice TC/WT/ET: {dice[0]:.3f} / {dice[1]:.3f} / {dice[2]:.3f}  "
        f"({mc_passes}-pass MC Dropout)",
        color="white", fontsize=11, fontweight="bold", y=0.98
    )

    plt.savefig(str(out_path), dpi=130, bbox_inches="tight", pad_inches=0.05,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ── Entropy Stats ────────────────────────────────────────────────────────────

def make_stats(subjects: list, out_path: Path):
    """
    Two panels:
      Left : violin plot — entropy distribution per region (TN/TP/FP/FN), one column per subject
      Right: bar chart — mean entropy per region across all subjects
    """
    BG = "black"
    REGION_COLORS = {
        "TN": "#555577",
        "TP": "#3dcc5e",
        "FP": "#ff7320",
        "FN": "#4477ff",
    }

    all_stats = {r: [] for r in ["TN", "TP", "FP", "FN"]}

    for subj in subjects:
        stats = entropy_stats(subj["entropy"], subj["gt"], subj["pred"])
        for region, vals in stats.items():
            all_stats[region].append(vals)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    ax.set_facecolor(BG)

    n = len(subjects)
    regions = ["TN", "TP", "FP", "FN"]
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width
    x_ticks = np.arange(n)

    for r_i, region in enumerate(regions):
        x_pos = x_ticks + offsets[r_i]
        data  = [all_stats[region][s_i] for s_i in range(n)]
        data = [d[::max(1, len(d) // 5000)] for d in data]

        parts = ax.violinplot(data, positions=x_pos, widths=width * 0.85,
                              showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(REGION_COLORS[region])
            pc.set_alpha(0.65)
            pc.set_edgecolor("none")
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.2)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        [Path(s["name"]).stem[-14:] for s in subjects],
        rotation=20, ha="right", fontsize=7.5, color="white"
    )
    ax.tick_params(colors="white", labelsize=8)
    ax.set_ylabel("Predictive Entropy", color="white", fontsize=10)
    ax.set_title("Entropy Distribution per Subject & Region", color="white",
                 fontsize=11, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_color("#555")
    ax.grid(axis="y", color="#333", linewidth=0.7)

    legend_handles = [mpatches.Patch(color=REGION_COLORS[r], label=r) for r in regions]
    ax.legend(handles=legend_handles, fontsize=9, labelcolor="white",
              facecolor=BG, edgecolor="#555", loc="upper right")

    ax = axes[1]
    ax.set_facecolor(BG)

    global_means = {}
    global_stds  = {}
    for region in regions:
        pooled = np.concatenate(all_stats[region])
        global_means[region] = float(pooled.mean())
        global_stds[region]  = float(pooled.std())

    x = np.arange(len(regions))
    means_arr = [global_means[r] for r in regions]
    stds_arr  = [global_stds[r]  for r in regions]
    colors    = [REGION_COLORS[r] for r in regions]

    bars = ax.bar(x, means_arr, yerr=stds_arr, color=colors,
                  width=0.5, capsize=5, error_kw={"ecolor": "white", "lw": 1.2},
                  edgecolor="none")

    for bar, mean_val in zip(bars, means_arr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{mean_val:.4f}", ha="center", va="bottom",
                color="white", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(regions, color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="white", labelsize=9)
    ax.set_ylabel("Mean Predictive Entropy (± std)", color="white", fontsize=10)
    ax.set_title(f"Entropy by Region — Aggregated over {n} subjects",
                 color="white", fontsize=11, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_color("#555")
    ax.grid(axis="y", color="#333", linewidth=0.7)

    ax.annotate(
        "Expected: TN < TP < FN ≤ FP\n(model most uncertain at missed & over-predicted regions)",
        xy=(0.5, 0.92), xycoords="axes fraction",
        ha="center", fontsize=8.5, color="#aaaacc",
        style="italic"
    )

    plt.tight_layout(pad=1.5)
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")
    return global_means, global_stds


# ── Calibration ──────────────────────────────────────────────────────────────

def collect_calibration_points(model, val_files, device, n_subjects, mc_passes,
                                subsample_per_subject, seed):
    """
    Runs MC inference over `n_subjects` and pools brain-mask voxels (subsampled per
    subject) into flat arrays: confidence (max softmax prob), correct (0/1, 4-class
    argmax vs GT), entropy (predictive entropy of the mean prediction).

    Restricting to brain-mask voxels (any modality > 0) avoids the dominant all-zero
    background trivially inflating both calibration and AUROC numbers.
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

        brain_mask = images_np.max(axis=0) > 0
        idx = np.flatnonzero(brain_mask)
        if idx.size > subsample_per_subject:
            idx = np.array(rng.sample(list(idx), subsample_per_subject))

        all_conf.append(conf_np.ravel()[idx])
        all_correct.append(correct_np.ravel()[idx])
        all_entropy.append(entropy_np.ravel()[idx])

    return (np.concatenate(all_conf), np.concatenate(all_correct),
            np.concatenate(all_entropy))


def expected_calibration_error(confidence, correct, n_bins=15):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(confidence, bin_edges[1:-1]), 0, n_bins - 1)

    bin_conf, bin_acc, bin_count = [], [], []
    ece = 0.0
    n_total = len(confidence)
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        bin_count.append(count)
        if count == 0:
            bin_conf.append(np.nan)
            bin_acc.append(np.nan)
            continue
        c = float(confidence[mask].mean())
        a = float(correct[mask].mean())
        bin_conf.append(c)
        bin_acc.append(a)
        ece += (count / n_total) * abs(a - c)

    return ece, np.array(bin_conf), np.array(bin_acc), np.array(bin_count), bin_edges


def make_calibration_plot(confidence, correct, entropy, out_path: Path):
    """
    Two panels:
      Left : reliability diagram (binned accuracy vs. confidence) + ECE
      Right: ROC curve — entropy as a predictor of voxel-level misclassification + AUROC
    """
    BG = "black"
    ece, bin_conf, bin_acc, bin_count, bin_edges = expected_calibration_error(confidence, correct)

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
    ax.set_title(f"Reliability Diagram  (ECE = {ece:.4f})", color="white",
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
    ax.set_title("Entropy as an Error Detector (ROC)", color="white",
                 fontsize=11, fontweight="bold")
    ax.tick_params(colors="white", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#555")
    ax.grid(color="#333", linewidth=0.7)
    ax.legend(fontsize=9, labelcolor="white", facecolor=BG, edgecolor="#555", loc="lower right")

    plt.tight_layout(pad=1.5)
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")
    return ece, auroc


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BraTS 2024 v2.1 (Attention U-Net) Uncertainty Visualization")
    p.add_argument("--checkpoint",     default="checkpoints_v2_1/best_model.pth")
    p.add_argument("--data_dir",       default="/home/sahar/CV_medical_data_project/processed/train")
    p.add_argument("--val_split_json", default=None,
                   help="Path to val_split.json (default: alongside checkpoint)")
    p.add_argument("--init_features",  type=int, default=32)
    p.add_argument("--out_dir",        default="uncertainty_vis_v2_1")
    p.add_argument("--mc_passes",      type=int, default=20)
    p.add_argument("--n_gallery_subjects", type=int, default=10)
    p.add_argument("--gallery_seed",   type=int, default=99)
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
    print(f"Val split: {len(val_files)} subjects ({split_path})")

    # ── Qualitative gallery ──────────────────────────────────────────────
    rng = random.Random(args.gallery_seed)
    sampled = rng.sample(val_files, min(args.n_gallery_subjects, len(val_files)))
    print(f"\nSampled {len(sampled)} subjects for gallery — {args.mc_passes} MC passes each")

    subjects = []
    for i, fpath in enumerate(sampled, 1):
        print(f"  [{i}/{len(sampled)}] {Path(fpath).name}")
        with h5py.File(fpath, "r") as f:
            images_np = f["images"][:]
            gt_np     = f["seg"][:].astype(np.int64)

        images_t = torch.from_numpy(images_np).unsqueeze(0).to(device)
        mean_pred, entropy_t = mc_inference(model, images_t, n_passes=args.mc_passes)
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

        subjects.append({
            "name": fpath, "images": images_np, "gt": gt_np,
            "pred": pred_class, "entropy": entropy_np, "dice": dice,
        })

    print("\nGenerating per-subject example figures...")
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    entropy_max_global = max(s["entropy"].max() for s in subjects)
    for i, subj in enumerate(subjects, 1):
        subj_name = Path(subj["name"]).stem
        make_subject_figure(
            subj, examples_dir / f"example_{i:02d}_{subj_name}.png",
            args.mc_passes, entropy_max_global,
        )

    print("Generating entropy stats figure...")
    global_means, global_stds = make_stats(subjects, out_dir / "uncertainty_stats.png")
    print(f"  Mean entropy by region: {global_means}")

    # ── Calibration analysis ─────────────────────────────────────────────
    print(f"\nRunning calibration analysis over {min(args.n_calib_subjects, len(val_files))} subjects...")
    confidence, correct, entropy = collect_calibration_points(
        model, val_files, device,
        n_subjects=args.n_calib_subjects, mc_passes=args.mc_passes,
        subsample_per_subject=args.calib_subsample_per_subject, seed=args.calib_seed,
    )
    ece, auroc = make_calibration_plot(confidence, correct, entropy, out_dir / "calibration.png")
    print(f"\nECE:   {ece:.4f}")
    print(f"AUROC: {auroc:.4f}  (entropy as voxel-error detector)")

    summary = {
        "checkpoint": args.checkpoint,
        "mc_passes": args.mc_passes,
        "n_gallery_subjects": len(subjects),
        "entropy_by_region_mean": global_means,
        "entropy_by_region_std": global_stds,
        "n_calib_subjects": min(args.n_calib_subjects, len(val_files)),
        "ece": ece,
        "auroc_entropy_error_detector": auroc,
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nDone. Output in {out_dir}/")


if __name__ == "__main__":
    main()
