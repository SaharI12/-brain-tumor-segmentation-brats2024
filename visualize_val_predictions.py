"""
Visualize model predictions on BraTS 2024 GLI validation subjects (no GT).
Loads:  processed/val/<subject>.h5  (preprocessed images)
        predictions/<subject>-seg.nii.gz  (predicted segmentation in original space)

For each subject: 3 axes × 3 panels (T1c | T1c+Pred | Pred only) + volume sidebar.
Outputs: exploration_output/val_predictions/subject_XX_<name>.png  (20 subjects)
"""

import random
from glob import glob
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# ── Config ───────────────────────────────────────────────────────────────────────
VAL_H5_DIR   = Path("processed/val")
PRED_DIR     = Path("predictions")
OUT_DIR      = Path("exploration_output/val_predictions")
N_SUBJECTS   = 20
VIZ_SEED     = 77

SEG_CMAP = mcolors.ListedColormap(["black", "red", "yellow", "cyan"])
SEG_NORM = mcolors.BoundaryNorm([0, 1, 2, 3, 4], SEG_CMAP.N)
BG = "#1a1a2e"

LABEL_INFO = {
    1: ("NCR",        "red",    "#e05c5c"),
    2: ("SNFH/Edema", "yellow", "#cccc44"),
    3: ("ET",         "cyan",   "#44cccc"),
}
REGION_INFO = [
    ("Whole Tumor (WT)", "#7777ee", lambda p: p > 0),
    ("Tumor Core  (TC)", "#e05c5c", lambda p: (p == 1) | (p == 3)),
    ("Enh. Tumor  (ET)", "#44cccc", lambda p: p == 3),
]
VOXEL_VOL_MM3 = 1.0   # 1 mm isotropic → 1 mm³ per voxel


# ── Helpers ───────────────────────────────────────────────────────────────────────

def normalize_slice(s: np.ndarray) -> np.ndarray:
    nz = s[s > 0]
    if nz.size == 0:
        return s
    p1, p99 = np.percentile(nz, [1, 99])
    return np.clip((s - p1) / (p99 - p1 + 1e-8), 0, 1)


def get_centroid(vol: np.ndarray) -> tuple:
    coords = np.argwhere(vol > 0)
    return tuple(coords.mean(axis=0).astype(int)) if len(coords) else tuple(s // 2 for s in vol.shape)


def get_slices(vol: np.ndarray, cx, cy, cz) -> dict:
    return {
        "Axial":    vol[:, :, cz].T,
        "Coronal":  vol[:, cy, :].T,
        "Sagittal": vol[cx, :, :].T,
    }


def cm3(n_voxels: int) -> float:
    return n_voxels * VOXEL_VOL_MM3 / 1000.0


# ── Per-subject figure ────────────────────────────────────────────────────────────

def make_subject_figure(subject: str, images_np: np.ndarray, pred_full: np.ndarray,
                        out_path: Path, idx: int, total: int):
    """
    images_np : (4, 160, 208, 160)  — preprocessed, in H5 crop space
    pred_full : (182, 218, 182)     — prediction in original NIfTI space
    We derive slices from pred_full and re-crop images to align for display.
    """
    # pred_full is in original (182,218,182); images are cropped (160,208,160).
    # For display, work entirely in the original space — reload t1c from H5 and
    # reverse-pad it to match pred_full isn't trivial, so instead we use the H5
    # images directly (already well-aligned to the prediction's brain region)
    # and slice the prediction in the same cropped space.

    # Get bounding box of prediction in original space → crop pred to match H5 shape
    # Strategy: use the brain mask from pred_full > 0 union of all non-zero to find
    # the bbox, then we work in the H5 (160,208,160) space for the images and crop
    # the pred back to that space using the same center_pad logic in reverse.
    # Simpler: just use the H5 images as-is and slice pred in H5 space too.
    # We stored images as (4,160,208,160); the pred in original space.
    # For display we just use the H5 prediction (re-run argmax from H5 would need model).
    # We'll project pred_full into the cropped space by finding its brain bbox and
    # using the same offsets as preprocess.

    # Find brain bbox in pred_full (non-zero region = tumor + where t1n>0, but we
    # only have the prediction here). Use the prediction's own extent for centroid.
    # For slicing, directly work in H5 space: need pred in H5 space.
    # We reconstruct: find bbox of pred_full non-zero, then infer the crop offsets.
    # Actually easier: we know preprocess tight-cropped by t1n>0 mask, but we don't
    # have t1n here. Instead we'll just display pred_full sliced at its own centroid.

    # Find centroid of pred_full tumor
    cx_f, cy_f, cz_f = get_centroid(pred_full)

    # Also get H5 image centroid (prediction projected into H5 space is approx same region)
    cx_h, cy_h, cz_h = get_centroid((pred_full > 0).astype(np.uint8))

    # Use pred_full for prediction display, and H5 t1c (index 1) for background MRI.
    # To align slices: display H5 at its own tumor centroid, pred_full at its own.
    # Since pred is in original space and images are in cropped space, we show them
    # separately but at their respective centroids (both are at the same anatomical location).
    t1c_h = images_np[1]   # (160, 208, 160) cropped, normalized

    # Centroid in H5 image space (use brightest tumor region in t1c)
    # If pred in original space has the tumor at (cx_f, cy_f, cz_f), we need the
    # equivalent in H5 space. Use the tumor region in pred_full to guide us.
    pred_h5 = _pred_to_h5_space(pred_full, t1c_h.shape)

    cx, cy, cz = get_centroid((pred_h5 > 0).astype(np.uint8))
    if (pred_h5 > 0).sum() == 0:
        cx, cy, cz = pred_h5.shape[0]//2, pred_h5.shape[1]//2, pred_h5.shape[2]//2

    axes_order = ["Axial", "Coronal", "Sagittal"]
    col_titles = ["T1c (MRI)", "T1c + Prediction", "Prediction Only"]

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.06, wspace=0.04,
                           width_ratios=[1, 1, 1, 0.80])

    t1c_slices  = get_slices(t1c_h,     cx, cy, cz)
    pred_slices = get_slices(pred_h5.astype(np.float32), cx, cy, cz)

    for row_i, axis_name in enumerate(axes_order):
        t1c_sl  = normalize_slice(t1c_slices[axis_name])
        pred_sl = pred_slices[axis_name]

        panels = []

        # T1c only
        ax = fig.add_subplot(gs[row_i, 0])
        ax.imshow(t1c_sl, cmap="gray", origin="lower")
        panels.append(ax)

        # T1c + Pred overlay
        ax = fig.add_subplot(gs[row_i, 1])
        ax.imshow(t1c_sl, cmap="gray", origin="lower")
        ax.imshow(pred_sl, cmap=SEG_CMAP, norm=SEG_NORM, origin="lower",
                  alpha=0.55, interpolation="nearest")
        panels.append(ax)

        # Pred colormap only (black background)
        ax = fig.add_subplot(gs[row_i, 2])
        ax.imshow(np.zeros_like(t1c_sl), cmap="gray", origin="lower")
        ax.imshow(pred_sl, cmap=SEG_CMAP, norm=SEG_NORM, origin="lower",
                  alpha=1.0, interpolation="nearest")
        panels.append(ax)

        for col_i, ax in enumerate(panels):
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#444")
            if row_i == 0:
                ax.set_title(col_titles[col_i], color="white", fontsize=10,
                             fontweight="bold", pad=5)
            if col_i == 0:
                ax.set_ylabel(axis_name, color="white", fontsize=10,
                              fontweight="bold", labelpad=6)

    # ── Sidebar: predicted volumes ────────────────────────────────────────────────
    ax_side = fig.add_subplot(gs[:, 3])
    ax_side.set_facecolor(BG)
    ax_side.axis("off")

    # Per-label voxel counts
    label_counts = {lbl: int((pred_full == lbl).sum()) for lbl in [1, 2, 3]}
    total_tumor  = int((pred_full > 0).sum())

    lines = [
        ("Predicted Volumes", None, "white", True),
        ("", None, "white", False),
        (f"NCR   (label 1)", label_counts[1], "#e05c5c", False),
        (f"SNFH  (label 2)", label_counts[2], "#cccc44", False),
        (f"ET    (label 3)", label_counts[3], "#44cccc", False),
        ("", None, "white", False),
        ("Sub-regions", None, "#aaaacc", True),
        (f"WT  = labels 1+2+3", total_tumor, "#7777ee", False),
        (f"TC  = labels 1+3",   label_counts[1] + label_counts[3], "#e05c5c", False),
        (f"ET  = label 3",      label_counts[3], "#44cccc", False),
    ]

    y = 0.96
    for label, count, color, bold in lines:
        if count is not None:
            text = f"{label}\n  {count:,} vox  ({cm3(count):.2f} cm³)"
        else:
            text = label
        ax_side.text(0.05, y, text, transform=ax_side.transAxes,
                     va="top", ha="left", fontsize=8.5,
                     color=color, fontweight="bold" if bold else "normal",
                     fontfamily="monospace")
        y -= 0.10 if count is not None else (0.04 if label else 0.03)

    # ── Legend ────────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="red",    label="NCR — Necrotic Core (1)"),
        mpatches.Patch(color="yellow", label="SNFH / Edema (2)"),
        mpatches.Patch(color="cyan",   label="Enhancing Tumor (3)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               fontsize=9, labelcolor="white", facecolor=BG,
               edgecolor="#555", framealpha=0.3, bbox_to_anchor=(0.46, -0.02))

    fig.suptitle(
        f"[{idx:02d}/{total}]  {subject}  |  BraTS 2024 GLI Validation (no GT)  |  "
        f"Tumor: {cm3(total_tumor):.1f} cm³   WT {cm3(total_tumor):.1f}  "
        f"TC {cm3(label_counts[1]+label_counts[3]):.1f}  ET {cm3(label_counts[3]):.1f} cm³",
        color="white", fontsize=10, fontweight="bold", y=1.01,
    )

    plt.savefig(str(out_path), dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


def _pred_to_h5_space(pred_full: np.ndarray, h5_shape: tuple) -> np.ndarray:
    """
    Project pred_full (182,218,182) into H5 crop space (160,208,160) by reversing
    the tight-crop + center_pad.  Uses the pred non-zero mask as a proxy for the
    brain mask (close enough for display purposes).
    """
    TARGET = (160, 208, 160)
    mask   = pred_full > 0

    # Tight bbox of the non-zero prediction region in original space
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.zeros(TARGET, dtype=pred_full.dtype)

    mins = coords.min(0); maxs = coords.max(0) + 1
    # Expand bbox a bit to approximate the full brain bbox (tumor is inside brain)
    # We add a generous margin since we only have the prediction, not the t1n brain mask
    margin = np.array([20, 20, 20])
    x0 = max(0, mins[0] - margin[0]); x1 = min(pred_full.shape[0], maxs[0] + margin[0])
    y0 = max(0, mins[1] - margin[1]); y1 = min(pred_full.shape[1], maxs[1] + margin[1])
    z0 = max(0, mins[2] - margin[2]); z1 = min(pred_full.shape[2], maxs[2] + margin[2])

    crop = pred_full[x0:x1, y0:y1, z0:z1]
    brain_shape = crop.shape

    # Reverse center_pad: extract TARGET-sized region from the padded brain
    result = np.zeros(TARGET, dtype=pred_full.dtype)
    slices_src, slices_dst = [], []
    for i in range(3):
        t, b = TARGET[i], brain_shape[i]
        if b > t:
            start = (b - t) // 2
            slices_src.append(slice(start, start + t))
            slices_dst.append(slice(0, t))
        else:
            offset = (t - b) // 2
            slices_src.append(slice(0, b))
            slices_dst.append(slice(offset, offset + b))
    result[tuple(slices_dst)] = crop[tuple(slices_src)]
    return result


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    h5_files   = sorted(VAL_H5_DIR.glob("*.h5"))
    pred_files = {p.name.replace("-seg.nii.gz", ""): p for p in PRED_DIR.glob("*-seg.nii.gz")}

    # Only keep subjects that have both H5 and prediction
    available = [f for f in h5_files if f.stem in pred_files]
    print(f"Available: {len(available)} subjects with both H5 and prediction")

    rng     = random.Random(VIZ_SEED)
    sampled = rng.sample(available, min(N_SUBJECTS, len(available)))
    print(f"Sampled: {len(sampled)} subjects\n")

    for i, h5_path in enumerate(sampled, 1):
        subject = h5_path.stem
        out_path = OUT_DIR / f"subject_{i:02d}_{subject}.png"
        print(f"  [{i:02d}/{len(sampled)}] {subject}")

        with h5py.File(str(h5_path), "r") as f:
            images_np = f["images"][:]    # (4, 160, 208, 160)

        pred_img  = nib.load(str(pred_files[subject]))
        pred_full = np.asarray(pred_img.dataobj).astype(np.uint8)  # (182, 218, 182)

        make_subject_figure(subject, images_np, pred_full, out_path, i, len(sampled))
        print(f"             → {out_path.name}")

    print(f"\nDone. {len(sampled)} figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()