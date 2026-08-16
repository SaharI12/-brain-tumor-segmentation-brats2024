"""
BraTS 2024 GLI v2.1 (Attention U-Net) — Multi-Subject Uncertainty GIFs

Runs MC Dropout inference on one or more subjects and renders each one's
predictive entropy two ways:
  1. 3D — a rotating point cloud over the whole brain (render_gif)
  2. 2D — an axial slice-by-slice sweep, each frame overlaying the entropy
     map on the T1c slice as an inferno glow (render_slice_gif)

Usage:
    # single subject
    python visualize_uncertainty_gif_v2.py [SUBJECT_ID] [--checkpoint PATH]
        [--data_dir DIR] [--mc_passes N] [--mode {3d,slices,both}]
        [--n_frames N] [--fps N] [--entropy_threshold T]
        [--z_step N] [--slice_fps N] [--init_features N]

    # N subjects sampled from the checkpoint's held-out val split
    python visualize_uncertainty_gif_v2.py --n_subjects 10

SUBJECT_ID is the .h5 filename stem, e.g. BraTS-GLI-00009-100 (default below).
Outputs (default: uncertainty_gif_v2_1/), one pair per subject:
  {SUBJECT_ID}_uncertainty_3d.gif
  {SUBJECT_ID}_uncertainty_axial_slices.gif
"""

import argparse
import io
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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
import imageio.v2 as imageio

from model_v2 import UNet3DAttn, mc_inference

DEFAULT_SUBJECT = "BraTS-GLI-00009-100"


def load_model(checkpoint, init_features, device):
    ckpt = torch.load(checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    model = UNet3DAttn(
        in_channels=4, out_channels=4,
        init_features=ckpt_args.get("init_features", init_features),
        dropout_p=ckpt_args.get("dropout", 0.15),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: epoch {ckpt.get('epoch', '?')}  "
          f"best Dice {ckpt.get('best_dice', float('nan')):.4f}")
    return model


def get_val_files(data_dir, split_path):
    with open(split_path) as f:
        saved = json.load(f)
    return [os.path.join(data_dir, fn) for fn in saved["val"]]


def load_subject_images(subject_id: str, data_dir: str) -> np.ndarray:
    path = os.path.join(data_dir, f"{subject_id}.h5")
    with h5py.File(path, "r") as f:
        images = f["images"][:]
    return images


def normalize_slice(s: np.ndarray) -> np.ndarray:
    nz = s[s > 0]
    if nz.size == 0:
        return s
    p1, p99 = np.percentile(nz, [1, 99])
    return np.clip((s - p1) / (p99 - p1 + 1e-8), 0, 1)


def axial(vol: np.ndarray, z: int) -> np.ndarray:
    return vol[:, :, z].T


def mesh_mask(mask: np.ndarray):
    """Mesh a binary mask with marching cubes. Returns (verts, faces) or None."""
    if mask.sum() < 8:
        return None
    try:
        verts, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5)
    except (ValueError, RuntimeError):
        return None
    return verts, faces


def add_mesh(ax, mesh, color, alpha, zorder=None):
    if mesh is None:
        return
    verts, faces = mesh
    poly = Poly3DCollection(verts[faces], alpha=alpha)
    poly.set_facecolor(color)
    poly.set_edgecolor("none")
    if zorder is not None:
        poly.set_zorder(zorder)
    ax.add_collection3d(poly)


def run_mc_inference(model, images_np, mc_passes, device):
    images_t = torch.from_numpy(images_np).unsqueeze(0).to(device)
    mean_pred, entropy_t = mc_inference(model, images_t, n_passes=mc_passes)
    pred_class = mean_pred.argmax(dim=1)[0].cpu().numpy()   # (X, Y, Z)
    entropy_np = entropy_t[0].cpu().numpy()                 # (X, Y, Z)
    return pred_class, entropy_np


def build_scene(images_np, pred_class, entropy_np, entropy_threshold):
    """Precompute everything that doesn't change between frames: brain mesh,
    tumor mesh, and the high-entropy point cloud (positions + colors)."""
    brain_mask = images_np[0] > 0  # t1n, skull-stripped
    brain_mesh = mesh_mask(brain_mask)

    tumor_mask = pred_class > 0  # predicted whole tumor (context only)
    tumor_mesh = mesh_mask(tumor_mask)

    # Confidently-classified background dominates the brain mask with entropy
    # ~1e-10 (not exactly 0, just numerically tiny) — a percentile-based cutoff
    # gets swamped by that mass, so use an absolute nats threshold instead:
    # it isolates the genuinely uncertain voxels (tumor boundaries) from noise.
    entropy_brain = np.where(brain_mask, entropy_np, 0.0)
    coords = np.argwhere(entropy_brain > entropy_threshold)
    vals = entropy_brain[entropy_brain > entropy_threshold]

    print(f"Entropy cloud: {len(coords):,} voxels above threshold={entropy_threshold} "
          f"nats (max={entropy_np.max():.4f})")

    return brain_mesh, tumor_mesh, coords, vals


def render_gif(subject_id, images_np, pred_class, entropy_np, entropy_threshold,
                n_frames, fps, out_path):
    brain_mesh, tumor_mesh, coords, vals = build_scene(
        images_np, pred_class, entropy_np, entropy_threshold
    )

    D, H, W = entropy_np.shape
    vmax = entropy_np.max()
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    point_colors = cm.inferno(norm(vals))

    BG = "#1a1a2e"
    fig = plt.figure(figsize=(8, 8))
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG)
    ax.computed_zorder = False  # draw order, not depth sort, controls compositing

    add_mesh(ax, brain_mesh, color="gray", alpha=0.035, zorder=1)
    add_mesh(ax, tumor_mesh, color="#5599ff", alpha=0.12, zorder=2)
    # depthshade=False: mplot3d's default fades points by camera distance,
    # which washes out the far side of the cloud regardless of the alpha we
    # set here — disabling it keeps every point at full, bold color.
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=point_colors, s=3, marker="o", linewidths=0, zorder=3,
               alpha=1.0, depthshade=False)

    ax.set_xlim(0, D)
    ax.set_ylim(0, H)
    ax.set_zlim(0, W)
    ax.set_box_aspect((D, H, W))
    ax.set_axis_off()

    cbar_ax = fig.add_axes((0.90, 0.15, 0.02, 0.7))
    sm = cm.ScalarMappable(cmap="inferno", norm=norm)
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Predictive Entropy", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    fig.suptitle(
        f"{subject_id} — MC Dropout Uncertainty (whole brain, Attention U-Net)\n"
        f"blue = predicted tumor region (context)",
        color="white", fontsize=11, y=0.97
    )

    frames = []
    elev0 = 15
    for i in range(n_frames):
        azim = 360.0 * i / n_frames
        elev = elev0 + 10 * np.sin(2 * np.pi * i / n_frames)
        ax.view_init(elev=elev, azim=azim)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()

        if (i + 1) % 10 == 0 or i + 1 == n_frames:
            print(f"  rendered frame {i + 1}/{n_frames}")

    plt.close(fig)

    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"Saved: {out_path}")


def render_slice_gif(subject_id, images_np, entropy_np, z_step, fps,
                      entropy_alpha_max, out_path):
    """
    Sweep through axial slices (array axis 2, matching axial() elsewhere in the
    repo). Each frame: T1c slice with the entropy map layered on top as a glow
    whose per-pixel alpha scales with entropy magnitude, so confidently
    classified voxels stay clean and only genuinely uncertain voxels light up.
    """
    brain_mask = images_np[0] > 0
    z_nonzero = np.where(brain_mask.any(axis=(0, 1)))[0]
    z_min, z_max = int(z_nonzero.min()), int(z_nonzero.max())
    z_range = list(range(z_min, z_max + 1, z_step))

    vmax = float(entropy_np.max())
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    BG = "black"
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor(BG)

    cbar_ax = fig.add_axes((0.90, 0.15, 0.02, 0.7))
    sm = cm.ScalarMappable(cmap="inferno", norm=norm)
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Predictive Entropy", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    frames = []
    for i, z in enumerate(z_range):
        ax.clear()
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([])

        t1c_sl = normalize_slice(axial(images_np[1], z))
        entropy_sl = axial(entropy_np, z)

        ax.imshow(t1c_sl, cmap="gray", origin="lower")

        # Entropy glow: alpha scales with entropy magnitude so it reads as an
        # overlay on the MRI rather than a flat, opaque heatmap. Square-root
        # curve (instead of linear) so mid-range entropy also shows up bold,
        # not just the single most-uncertain voxel.
        entropy_rgba = cm.inferno(norm(entropy_sl))
        entropy_frac = np.clip(entropy_sl / max(vmax, 1e-8), 0, 1)
        entropy_rgba[..., 3] = np.sqrt(entropy_frac) * entropy_alpha_max
        ax.imshow(entropy_rgba, origin="lower", interpolation="bilinear")

        ax.set_title(f"{subject_id} — axial slice z={z} ({i + 1}/{len(z_range)})\n"
                     f"MC Dropout predictive entropy",
                     color="white", fontsize=10)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()

        if (i + 1) % 25 == 0 or i + 1 == len(z_range):
            print(f"  rendered slice {i + 1}/{len(z_range)} (z={z})")

    plt.close(fig)

    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"Saved: {out_path}")


def process_subject(subject_id, model, args, out_dir, device):
    print(f"\n=== {subject_id} ===")
    images_np = load_subject_images(subject_id, args.data_dir)
    print(f"images shape: {images_np.shape}")

    pred_class, entropy_np = run_mc_inference(model, images_np, args.mc_passes, device)

    if args.mode in ("3d", "both"):
        out_path = out_dir / f"{subject_id}_uncertainty_3d.gif"
        render_gif(
            subject_id, images_np, pred_class, entropy_np,
            args.entropy_threshold, args.n_frames, args.fps, out_path
        )

    if args.mode in ("slices", "both"):
        out_path = out_dir / f"{subject_id}_uncertainty_axial_slices.gif"
        render_slice_gif(
            subject_id, images_np, entropy_np,
            args.z_step, args.slice_fps, args.entropy_alpha_max, out_path
        )


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-subject uncertainty GIFs (v2.1 Attention U-Net, MC Dropout)"
    )
    p.add_argument("subject", nargs="?", default=DEFAULT_SUBJECT,
                    help=f"Subject .h5 stem (default: {DEFAULT_SUBJECT}); ignored if --n_subjects > 1")
    p.add_argument("--n_subjects", type=int, default=1,
                    help="If > 1, sample this many subjects from the checkpoint's held-out "
                         "val split (--val_split_json) instead of using the positional subject")
    p.add_argument("--sample_seed", type=int, default=99)
    p.add_argument("--val_split_json", default=None,
                    help="Path to val_split.json (default: alongside checkpoint)")
    p.add_argument("--checkpoint", default="checkpoints_v2_1/best_model.pth")
    p.add_argument("--data_dir", default="/home/sahar/CV_medical_data_project/processed/train")
    p.add_argument("--mc_passes", type=int, default=20)
    p.add_argument("--mode", choices=["3d", "slices", "both"], default="both",
                    help="3d = rotating point cloud, slices = axial slice sweep, both = default")
    p.add_argument("--n_frames", type=int, default=60, help="Frames per 360° rotation (3d mode)")
    p.add_argument("--fps", type=int, default=15, help="FPS for the 3d rotation GIF")
    p.add_argument("--entropy_threshold", type=float, default=0.05,
                    help="3d mode: only render voxels with entropy above this (nats); "
                         "~0.05 isolates tumor-boundary uncertainty from background noise")
    p.add_argument("--z_step", type=int, default=1,
                    help="slices mode: step between axial slices (1 = every slice)")
    p.add_argument("--slice_fps", type=int, default=4,
                    help="FPS for the axial slice GIF (kept low so each slice is visible)")
    p.add_argument("--entropy_alpha_max", type=float, default=1.0,
                    help="slices mode: max opacity of the entropy glow overlay")
    p.add_argument("--init_features", type=int, default=32)
    p.add_argument("--out_dir", default="uncertainty_gif_v2_1")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, args.init_features, device)

    if args.n_subjects > 1:
        split_path = args.val_split_json or os.path.join(os.path.dirname(args.checkpoint), "val_split.json")
        val_files = get_val_files(args.data_dir, split_path)
        rng = random.Random(args.sample_seed)
        sampled = rng.sample(val_files, min(args.n_subjects, len(val_files)))
        subject_ids = [Path(f).stem for f in sampled]
        print(f"Sampled {len(subject_ids)} subjects from val split "
              f"(seed={args.sample_seed}, {split_path}): {subject_ids}")
    else:
        subject_ids = [args.subject]

    for subject_id in subject_ids:
        process_subject(subject_id, model, args, out_dir, device)


if __name__ == "__main__":
    main()