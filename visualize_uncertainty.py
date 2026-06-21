"""
Uncertainty visualization: MC Dropout entropy maps overlaid on MRI.
Outputs to exploration_output/uncertainty_vis/
  - uncertainty_gallery.png   : 6 subjects × 5 panels (axial at tumor centroid)
  - uncertainty_stats.png     : entropy distribution split by TP / FP / FN / TN
"""

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
from matplotlib.colorbar import ColorbarBase

from model import UNet3D, mc_inference, get_region_masks


# ── Config ───────────────────────────────────────────────────────────────────────
OUT_DIR    = Path("exploration_output/uncertainty_vis")
N_SUBJECTS = 6
MC_PASSES  = 20
VIZ_SEED   = 99

SEG_CMAP = mcolors.ListedColormap(["black", "red", "yellow", "cyan"])
SEG_NORM = mcolors.BoundaryNorm([0, 1, 2, 3, 4], SEG_CMAP.N)


# ── Helpers ───────────────────────────────────────────────────────────────────────

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


def axial(vol, z):
    return vol[:, :, z].T


def entropy_stats(entropy_np: np.ndarray, gt: np.ndarray, pred: np.ndarray):
    """Return mean entropy per region: TN, TP, FP, FN (whole-tumor binary)."""
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


# ── Gallery ───────────────────────────────────────────────────────────────────────

def make_gallery(subjects: list, out_path: Path):
    """
    5 panels per subject (axial, tumor centroid):
      T1c  |  T1c + GT  |  T1c + MC Pred  |  Entropy map  |  T1c + Entropy overlay
    """
    n = len(subjects)
    BG = "#1a1a2e"
    col_titles = ["T1c (MRI)", "T1c + GT", "T1c + MC Pred", f"Entropy\n({MC_PASSES} passes)", "T1c + Entropy"]

    fig = plt.figure(figsize=(22, 3.5 * n + 0.6))
    fig.patch.set_facecolor(BG)
    outer = gridspec.GridSpec(n, 1, figure=fig, hspace=0.10)

    entropy_max_global = max(s["entropy"].max() for s in subjects)

    for row_i, subj in enumerate(subjects):
        imgs     = subj["images"]     # (4, X, Y, Z)
        gt       = subj["gt"]         # (X, Y, Z)
        pred     = subj["pred"]       # (X, Y, Z)
        entropy  = subj["entropy"]    # (X, Y, Z)
        dice     = subj["dice"]
        name     = Path(subj["name"]).stem

        cx, cy, cz = get_centroid(gt)

        t1c_sl  = normalize_slice(axial(imgs[1], cz))
        gt_sl   = axial(gt.astype(np.float32), cz)
        pred_sl = axial(pred.astype(np.float32), cz)
        ent_sl  = axial(entropy, cz)

        inner = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=outer[row_i], wspace=0.04
        )

        panels = []

        # 0: T1c only
        ax = fig.add_subplot(inner[0, 0])
        ax.imshow(t1c_sl, cmap="gray", origin="lower")
        panels.append(ax)

        # 1: T1c + GT
        ax = fig.add_subplot(inner[0, 1])
        ax.imshow(t1c_sl, cmap="gray", origin="lower")
        ax.imshow(gt_sl, cmap=SEG_CMAP, norm=SEG_NORM, origin="lower",
                  alpha=0.55, interpolation="nearest")
        panels.append(ax)

        # 2: T1c + MC Pred
        ax = fig.add_subplot(inner[0, 2])
        ax.imshow(t1c_sl, cmap="gray", origin="lower")
        ax.imshow(pred_sl, cmap=SEG_CMAP, norm=SEG_NORM, origin="lower",
                  alpha=0.55, interpolation="nearest")
        panels.append(ax)

        # 3: Entropy map (standalone, "inferno")
        ax = fig.add_subplot(inner[0, 3])
        ax.imshow(np.zeros_like(t1c_sl), cmap="gray", origin="lower")
        im = ax.imshow(ent_sl, cmap="inferno", origin="lower",
                       vmin=0, vmax=entropy_max_global, interpolation="bilinear")
        panels.append(ax)

        # 4: T1c + Entropy overlay (only where uncertain: ent > 10th percentile of non-zero)
        ax = fig.add_subplot(inner[0, 4])
        ax.imshow(t1c_sl, cmap="gray", origin="lower")
        # Mask entropy to tumor region (union of GT and pred) for cleaner overlay
        tumor_mask_sl = axial(((gt > 0) | (pred > 0)).astype(np.float32), cz)
        ent_masked = ent_sl * (tumor_mask_sl > 0)
        ax.imshow(ent_masked, cmap="inferno", origin="lower",
                  vmin=0, vmax=entropy_max_global, alpha=0.75, interpolation="bilinear")
        panels.append(ax)

        for col_j, ax in enumerate(panels):
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#444")
            if row_i == 0:
                ax.set_title(col_titles[col_j], color="white", fontsize=9,
                             fontweight="bold", pad=4)

        # Row label: subject name + Dice
        panels[0].set_ylabel(
            f"{name[-20:]}\nDice {dice[0]:.3f}/{dice[1]:.3f}/{dice[2]:.3f}",
            fontsize=7, color="white", rotation=0,
            labelpad=4, ha="right", va="center"
        )

    # Shared colorbar for entropy
    cbar_ax = fig.add_axes([0.92, 0.12, 0.012, 0.75])
    norm = mcolors.Normalize(vmin=0, vmax=entropy_max_global)
    cb = ColorbarBase(cbar_ax, cmap=plt.cm.inferno, norm=norm, orientation="vertical")
    cb.set_label("Predictive Entropy", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    cb.outline.set_edgecolor("#555")

    # Legend
    legend_entries = [
        mpatches.Patch(color="red",    label="NCR (1)"),
        mpatches.Patch(color="yellow", label="SNFH/Edema (2)"),
        mpatches.Patch(color="cyan",   label="ET (3)"),
    ]
    fig.legend(handles=legend_entries, loc="lower center", ncol=3,
               fontsize=8.5, labelcolor="white", facecolor=BG,
               edgecolor="#555", framealpha=0.3,
               bbox_to_anchor=(0.46, -0.01))

    fig.suptitle(
        f"BraTS 2024 GLI — MC Dropout Uncertainty  ({MC_PASSES} passes)  |  "
        f"Dice label: TC / WT / ET",
        color="white", fontsize=11, fontweight="bold", y=1.002
    )

    plt.savefig(str(out_path), dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ── Entropy Stats ─────────────────────────────────────────────────────────────────

def make_stats(subjects: list, out_path: Path):
    """
    Two panels:
      Left : violin plot — entropy distribution per region (TN/TP/FP/FN), one column per subject
      Right: bar chart — mean entropy per region across all subjects
    """
    BG = "#1a1a2e"
    REGION_COLORS = {
        "TN": "#555577",
        "TP": "#3dcc5e",
        "FP": "#ff7320",
        "FN": "#4477ff",
    }

    # Collect entropy values per region across all subjects
    all_stats = {r: [] for r in ["TN", "TP", "FP", "FN"]}
    per_subject_means = []   # list of dicts {region: mean}

    for subj in subjects:
        stats = entropy_stats(subj["entropy"], subj["gt"], subj["pred"])
        means = {}
        for region, vals in stats.items():
            all_stats[region].append(vals)
            means[region] = float(vals.mean())
        per_subject_means.append(means)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(BG)

    # ── Left: violin per subject ──────────────────────────────────────────────
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
        # Sub-sample large TN region for violin speed
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
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color("#555")
    ax.grid(axis="y", color="#333", linewidth=0.7)

    legend_handles = [mpatches.Patch(color=REGION_COLORS[r], label=r) for r in regions]
    ax.legend(handles=legend_handles, fontsize=9, labelcolor="white",
              facecolor=BG, edgecolor="#555", loc="upper right")

    # ── Right: mean entropy per region (bar) ─────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(BG)

    # Compute global mean and std per region
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

    # Annotation: expected ordering
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


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt      = torch.load("checkpoints/best_model.pth", map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = UNet3D(
        in_channels=4, out_channels=4,
        init_features=ckpt_args.get("init_features", 32),
        dropout_p=ckpt_args.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded: epoch {ckpt.get('epoch')}  best Dice {ckpt.get('best_dice'):.4f}")

    val_files = get_val_files("processed/train", val_split=0.2, seed=42)
    rng       = random.Random(VIZ_SEED)
    sampled   = rng.sample(val_files, min(N_SUBJECTS, len(val_files)))
    print(f"\nSampled {len(sampled)} subjects — running {MC_PASSES} MC passes each")

    subjects = []
    for i, fpath in enumerate(sampled, 1):
        print(f"  [{i}/{len(sampled)}] {Path(fpath).name}")
        with h5py.File(fpath, "r") as f:
            images_np = f["images"][:]
            gt_np     = f["seg"][:].astype(np.int64)

        images_t = torch.from_numpy(images_np).unsqueeze(0).to(device)

        mean_pred, entropy_t = mc_inference(model, images_t, n_passes=MC_PASSES)
        pred_class = mean_pred.argmax(dim=1)[0].cpu().numpy()   # (X, Y, Z)
        entropy_np = entropy_t[0].cpu().numpy()                  # (X, Y, Z)

        # per-subject mean Dice (TC, WT, ET)
        def dice_bin(a, b):
            i = (a & b).sum(); d = a.sum() + b.sum()
            return 2 * i / d if d > 0 else 1.0
        dice = np.array([
            dice_bin((pred_class == 1) | (pred_class == 3), (gt_np == 1) | (gt_np == 3)),
            dice_bin( pred_class > 0,  gt_np > 0),
            dice_bin( pred_class == 3, gt_np == 3),
        ])

        subjects.append({
            "name":    fpath,
            "images":  images_np,
            "gt":      gt_np,
            "pred":    pred_class,
            "entropy": entropy_np,
            "dice":    dice,
        })

    print("\nGenerating uncertainty gallery...")
    make_gallery(subjects, OUT_DIR / "uncertainty_gallery.png")

    print("Generating entropy stats figure...")
    make_stats(subjects, OUT_DIR / "uncertainty_stats.png")

    print(f"\nDone. Output in {OUT_DIR}/")


if __name__ == "__main__":
    main()