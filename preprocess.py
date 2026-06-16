"""
BraTS 2024 GLI — Preprocessing Script

For each subject:
  1. Load 4 modalities + seg with nibabel
  2. Normalize each modality by its global max (computed across all training subjects)
  3. Remap label 4 → 3  (both mean ET; standardize to {0, 1, 2, 3})
  4. Tight-crop to brain bounding box (remove all-zero borders using t1n mask)
  5. Center-pad to fixed size 160 × 208 × 160
  6. Save as individual .h5 file: images (4,160,208,160) float32 + seg (160,208,160) uint8

Output layout:
  processed/
    train/  →  1,621 .h5 files  (TrainingData + AdditionalTrainingData)
    val/    →    188 .h5 files  (ValidationData, images only — no seg)
    normalization_stats.json
"""

import json
import numpy as np
import nibabel as nib
import h5py
from pathlib import Path
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT       = Path("/home/sahar/CV_medical_data_project")
TRAIN_DIR  = ROOT / "BraTS2024-BraTS-GLI-TrainingData"  / "training_data1_v2"
EXTRA_DIR  = ROOT / "BraTS2024-BraTS-GLI-AdditionalTrainingData" / "training_data_additional"
VAL_DIR    = ROOT / "BraTS2024-BraTS-GLI-ValidationData" / "validation_data"

OUT_ROOT   = ROOT / "processed"
OUT_TRAIN  = OUT_ROOT / "train"
OUT_VAL    = OUT_ROOT / "val"
STATS_FILE = OUT_ROOT / "normalization_stats.json"

MODALITIES  = ["t1n", "t1c", "t2w", "t2f"]
TARGET_SIZE = (160, 208, 160)   # fixed output shape (X, Y, Z) — divisible by 16


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_volume(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj)


def get_subject_paths(subject_dir: Path) -> dict:
    name = subject_dir.name
    paths = {mod: subject_dir / f"{name}-{mod}.nii.gz" for mod in MODALITIES}
    seg_path = subject_dir / f"{name}-seg.nii.gz"
    paths["seg"] = seg_path if seg_path.exists() else None
    return paths


def tight_crop_bbox(mask: np.ndarray):
    """Return (x0,x1, y0,y1, z0,z1) tight bounding box of non-zero voxels."""
    x = np.any(mask, axis=(1, 2))
    y = np.any(mask, axis=(0, 2))
    z = np.any(mask, axis=(0, 1))
    x0, x1 = np.where(x)[0][[0, -1]]
    y0, y1 = np.where(y)[0][[0, -1]]
    z0, z1 = np.where(z)[0][[0, -1]]
    return (x0, x1 + 1), (y0, y1 + 1), (z0, z1 + 1)


def center_pad(vol: np.ndarray, target: tuple) -> np.ndarray:
    """Fit vol to target shape: center-crop any dim that is too large, zero-pad any dim that is too small."""
    # Step 1: center-crop dimensions that exceed target
    slices = []
    for i in range(3):
        if vol.shape[i] > target[i]:
            start = (vol.shape[i] - target[i]) // 2
            slices.append(slice(start, start + target[i]))
        else:
            slices.append(slice(None))
    vol = vol[tuple(slices)]

    # Step 2: zero-pad dimensions that are smaller than target
    result = np.zeros(target, dtype=vol.dtype)
    offsets = [(target[i] - vol.shape[i]) // 2 for i in range(3)]
    slices_dst = tuple(slice(o, o + vol.shape[i]) for i, o in enumerate(offsets))
    result[slices_dst] = vol
    return result


# ── Step 1: Find global max per modality across all training subjects ──────────

def compute_global_stats(subject_dirs: list[Path]) -> dict:
    print("Step 1 — Computing global max per modality over training subjects...")
    global_max = {mod: 0.0 for mod in MODALITIES}

    for subj in tqdm(subject_dirs, desc="scanning"):
        paths = get_subject_paths(subj)
        for mod in MODALITIES:
            vol = load_volume(paths[mod])
            m = float(vol.max())
            if m > global_max[mod]:
                global_max[mod] = m

    print("\nGlobal max per modality:")
    for mod, val in global_max.items():
        print(f"  {mod}: {val:.2f}")
    return global_max


# ── Step 2: Process and save a single subject ──────────────────────────────────

def process_subject(subj_dir: Path, global_max: dict, out_dir: Path):
    paths = get_subject_paths(subj_dir)

    # Load modalities
    vols = {mod: load_volume(paths[mod]).astype(np.float32) for mod in MODALITIES}

    # Brain mask from t1n (skull-stripped → non-zero = brain)
    brain_mask = vols["t1n"] > 0
    if not brain_mask.any():
        print(f"  WARNING: {subj_dir.name} has no non-zero t1n voxels — skipping")
        return

    # Tight crop
    (x0, x1), (y0, y1), (z0, z1) = tight_crop_bbox(brain_mask)

    # Normalize + crop + pad each modality
    images = []
    for mod in MODALITIES:
        vol = vols[mod]
        vol = vol / global_max[mod]                   # → [0, 1]
        vol = vol[x0:x1, y0:y1, z0:z1]               # tight crop
        vol = center_pad(vol, TARGET_SIZE)             # pad to fixed size
        images.append(vol)

    images = np.stack(images, axis=0)                 # (4, 160, 208, 160) float32

    # Segmentation
    seg = None
    if paths["seg"] is not None:
        seg = load_volume(paths["seg"]).astype(np.uint8)
        seg[seg == 4] = 3                             # merge ET labels
        seg = seg[x0:x1, y0:y1, z0:z1]
        seg = center_pad(seg, TARGET_SIZE)            # (160, 208, 160) uint8

    # Save
    out_path = out_dir / f"{subj_dir.name}.h5"
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("images", data=images, compression="gzip", compression_opts=4)
        if seg is not None:
            f.create_dataset("seg", data=seg, compression="gzip", compression_opts=4)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT_TRAIN.mkdir(parents=True, exist_ok=True)
    OUT_VAL.mkdir(parents=True, exist_ok=True)

    train_subjects = sorted(TRAIN_DIR.iterdir()) + sorted(EXTRA_DIR.iterdir())
    val_subjects   = sorted(VAL_DIR.iterdir())

    print(f"Training subjects : {len(train_subjects)}")
    print(f"Validation subjects: {len(val_subjects)}")

    # Step 1 — global normalization stats (training only)
    if STATS_FILE.exists():
        print(f"\nLoading existing stats from {STATS_FILE}")
        with open(STATS_FILE) as f:
            global_max = json.load(f)
    else:
        global_max = compute_global_stats(train_subjects)
        with open(STATS_FILE, "w") as f:
            json.dump(global_max, f, indent=2)
        print(f"Saved stats to {STATS_FILE}")

    # Step 2 — process training subjects
    print(f"\nStep 2 — Processing {len(train_subjects)} training subjects...")
    skipped = []
    for subj in tqdm(train_subjects, desc="train"):
        out_path = OUT_TRAIN / f"{subj.name}.h5"
        if out_path.exists():
            continue  # resume-safe: skip already-processed subjects
        try:
            process_subject(subj, global_max, OUT_TRAIN)
        except Exception as e:
            print(f"  ERROR {subj.name}: {e}")
            skipped.append(subj.name)

    # Step 3 — process validation subjects (no seg)
    print(f"\nStep 3 — Processing {len(val_subjects)} validation subjects...")
    for subj in tqdm(val_subjects, desc="val"):
        out_path = OUT_VAL / f"{subj.name}.h5"
        if out_path.exists():
            continue
        try:
            process_subject(subj, global_max, OUT_VAL)
        except Exception as e:
            print(f"  ERROR {subj.name}: {e}")
            skipped.append(subj.name)

    print("\n" + "=" * 50)
    print(f"Done. Train files : {len(list(OUT_TRAIN.glob('*.h5')))}")
    print(f"      Val files   : {len(list(OUT_VAL.glob('*.h5')))}")
    if skipped:
        print(f"      Skipped     : {skipped}")
    print(f"Output: {OUT_ROOT}")
