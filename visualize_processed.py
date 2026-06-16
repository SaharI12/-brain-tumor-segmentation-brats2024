"""
Visualize preprocessed H5 files.
For each subject: 5 rows (t1n, t1c, t2w, t2f, seg) × 3 columns (axial, coronal, sagittal).
Slices are taken at the tumor centroid so the mask is always visible.
Saves 10 PNGs to exploration_output/processed_vis/.
"""

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import random

# ── Config ─────────────────────────────────────────────────────────────────────
PROCESSED_TRAIN = Path("/home/sahar/CV_medical_data_project/processed/train")
OUT_DIR         = Path("/home/sahar/CV_medical_data_project/exploration_output/processed_vis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODALITY_NAMES  = ["T1n", "T1c", "T2w", "T2f"]
N_EXAMPLES      = 10

# Seg colormap: 0=black, 1=red (NCR), 2=yellow (SNFH), 3=cyan (ET)
SEG_CMAP = mcolors.ListedColormap(["black", "red", "yellow", "cyan"])
SEG_NORM = mcolors.BoundaryNorm([0, 1, 2, 3, 4], SEG_CMAP.N)


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_centroid(seg: np.ndarray) -> tuple:
    """Return (x, y, z) centroid of tumor; fall back to volume center if no tumor."""
    coords = np.argwhere(seg > 0)
    if len(coords) > 0:
        return tuple(coords.mean(axis=0).astype(int))
    return tuple(s // 2 for s in seg.shape)


def normalize_slice(s: np.ndarray) -> np.ndarray:
    """Clip to [p1, p99] of non-zero values then scale to [0, 1]."""
    nz = s[s > 0]
    if nz.size == 0:
        return s
    p1, p99 = np.percentile(nz, [1, 99])
    return np.clip((s - p1) / (p99 - p1 + 1e-8), 0, 1)


def visualize_subject(h5_path: Path, out_path: Path):
    with h5py.File(str(h5_path), "r") as f:
        images = f["images"][:]   # (4, 160, 208, 160)
        seg    = f["seg"][:]      # (160, 208, 160)

    cx, cy, cz = get_centroid(seg)

    # Slices for each axis at centroid
    # images[mod] has shape (X, Y, Z)
    def slices(vol):
        return {
            "Axial"    : vol[:, :, cz].T,    # → (Y, X) displayed with origin lower
            "Coronal"  : vol[:, cy, :].T,    # → (Z, X)
            "Sagittal" : vol[cx, :, :].T,    # → (Z, Y)
        }

    col_labels  = MODALITY_NAMES + ["Seg", "T1c + Seg"]
    axis_labels = ["Axial", "Coronal", "Sagittal"]

    fig, axes = plt.subplots(3, 6, figsize=(21, 11))
    fig.suptitle(h5_path.stem, fontsize=13, y=0.995)

    for row, axis_name in enumerate(axis_labels):
        # Modality columns
        for col in range(4):
            ax = axes[row, col]
            s = slices(images[col])[axis_name]
            ax.imshow(normalize_slice(s), cmap="gray", origin="lower")
            if row == 0:
                ax.set_title(col_labels[col], fontsize=11)
            ax.set_ylabel(axis_name if col == 0 else "", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

        # Seg column
        ax = axes[row, 4]
        s = slices(seg.astype(np.float32))[axis_name]
        ax.imshow(s, cmap=SEG_CMAP, norm=SEG_NORM, origin="lower", interpolation="nearest")
        if row == 0:
            ax.set_title("Seg", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

        # T1c + Seg overlay column
        ax = axes[row, 5]
        t1c_s = slices(images[1])[axis_name]          # T1c is index 1
        seg_s = slices(seg.astype(np.float32))[axis_name]
        ax.imshow(normalize_slice(t1c_s), cmap="gray", origin="lower")
        ax.imshow(seg_s, cmap=SEG_CMAP, norm=SEG_NORM, origin="lower",
                  interpolation="nearest", alpha=0.45)
        if row == 0:
            ax.set_title("T1c + Seg", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    # Side legend explaining seg colors
    import matplotlib.patches as mpatches
    legend_entries = [
        ("black",  "0 — Background",    "Non-brain / outside skull"),
        ("red",    "1 — NCR",           "Necrotic Tumor Core\n(dead tissue at tumor center)"),
        ("yellow", "2 — SNFH / Edema",  "Surrounding Non-Enhancing\nFLAIR Hyperintensity\n(swollen tissue around tumor)"),
        ("cyan",   "3 — ET",            "Enhancing Tumor\n(active tumor, lights up\nwith contrast agent in T1c)"),
    ]
    handles = [
        mpatches.Patch(facecolor=color, edgecolor="gray", linewidth=0.5,
                       label=f"{label}\n{desc}")
        for color, label, desc in legend_entries
    ]
    fig.legend(handles=handles, loc="center right", fontsize=8.5,
               framealpha=0.95, edgecolor="gray", handlelength=2,
               handleheight=2, borderpad=1.0, labelspacing=1.4,
               title="Segmentation Labels", title_fontsize=9)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout(rect=[0, 0, 0.80, 0.995])
    plt.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_files = sorted(PROCESSED_TRAIN.glob("*.h5"))
    random.seed(42)
    sample = random.sample(all_files, N_EXAMPLES)

    for i, h5_path in enumerate(sample, 1):
        out_path = OUT_DIR / f"{i:02d}_{h5_path.stem}.png"
        print(f"[{i}/{N_EXAMPLES}] {h5_path.stem} → {out_path.name}")
        visualize_subject(h5_path, out_path)

    print(f"\nDone. Saved to {OUT_DIR}/")
