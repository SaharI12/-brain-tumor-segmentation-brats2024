"""
BraTS 2024 GLI — Generate submission predictions for the 188 validation subjects.

Workflow per subject:
  1. Load raw t1n NIfTI → original shape (182,218,182), affine, brain bbox
  2. Load preprocessed H5 from processed/val/
  3. Run model inference → prediction in (160,208,160) space
  4. Reverse center_pad → brain-crop-sized prediction
  5. Place into (182,218,182) zeros using bbox → full-volume prediction
  6. Save as <subject>-seg.nii.gz with original affine + header

Output: predictions/<subject>-seg.nii.gz  (+ predictions.zip for upload)
Labels: 0=background, 1=NCR, 2=SNFH/edema, 3=ET  (BraTS 2024 convention)
"""

import argparse
import zipfile
from glob import glob
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from model import UNet3D, mc_inference


# ── Config ───────────────────────────────────────────────────────────────────────
VAL_H5_DIR  = Path("processed/val")
VAL_NII_DIR = Path("BraTS2024-BraTS-GLI-ValidationData/validation_data")
OUT_DIR     = Path("predictions")
TARGET_SIZE = (160, 208, 160)


# ── Preprocessing reversal ────────────────────────────────────────────────────────

def tight_crop_bbox(mask: np.ndarray):
    x = np.any(mask, axis=(1, 2)); y = np.any(mask, axis=(0, 2)); z = np.any(mask, axis=(0, 1))
    x0, x1 = np.where(x)[0][[0, -1]]; y0, y1 = np.where(y)[0][[0, -1]]; z0, z1 = np.where(z)[0][[0, -1]]
    return (int(x0), int(x1) + 1), (int(y0), int(y1) + 1), (int(z0), int(z1) + 1)


def reverse_center_pad(pred_padded: np.ndarray, brain_shape: tuple) -> np.ndarray:
    """
    Invert center_pad: extract the brain-crop-sized prediction from the
    padded (160,208,160) prediction volume.

    center_pad logic:
      - if brain_dim > target_dim: center-cropped  → pred contains center of brain
      - if brain_dim < target_dim: zero-padded     → brain is at offset inside pred
    """
    result = np.zeros(brain_shape, dtype=pred_padded.dtype)
    slices_src = []   # from padded prediction
    slices_dst = []   # into brain-shaped output

    for i in range(3):
        t = TARGET_SIZE[i]
        b = brain_shape[i]
        if b > t:
            # preprocess center-cropped: pred covers center of original brain
            start = (b - t) // 2
            slices_src.append(slice(0, t))
            slices_dst.append(slice(start, start + t))
        else:
            # preprocess zero-padded: brain sits at offset inside pred
            offset = (t - b) // 2
            slices_src.append(slice(offset, offset + b))
            slices_dst.append(slice(0, b))

    result[tuple(slices_dst)] = pred_padded[tuple(slices_src)]
    return result


# ── Main ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default="checkpoints/best_model.pth")
    p.add_argument("--init_features", type=int, default=32)
    p.add_argument("--mc_passes",     type=int, default=0,
                   help="MC Dropout passes (0 = deterministic, recommended for submission)")
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
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
    print(f"Inference mode: {'MC Dropout ×' + str(args.mc_passes) if args.mc_passes > 0 else 'deterministic'}\n")

    h5_files = sorted(VAL_H5_DIR.glob("*.h5"))
    print(f"Processing {len(h5_files)} validation subjects...\n")

    for h5_path in tqdm(h5_files, desc="predicting"):
        subject = h5_path.stem   # e.g. BraTS-GLI-02073-100
        out_nii = OUT_DIR / f"{subject}-seg.nii.gz"
        if out_nii.exists():
            continue

        # ── 1. Load raw t1n NIfTI → affine, shape, bbox ──────────────────────
        t1n_path = VAL_NII_DIR / subject / f"{subject}-t1n.nii.gz"
        if not t1n_path.exists():
            print(f"  WARNING: raw NIfTI not found for {subject}, skipping")
            continue

        ref_img   = nib.load(str(t1n_path))
        orig_shape = ref_img.shape          # (182, 218, 182)
        affine    = ref_img.affine
        header    = ref_img.header

        t1n_arr   = np.asarray(ref_img.dataobj)
        brain_mask = t1n_arr > 0
        (x0, x1), (y0, y1), (z0, z1) = tight_crop_bbox(brain_mask)
        brain_shape = (x1 - x0, y1 - y0, z1 - z0)

        # ── 2. Load preprocessed H5 ───────────────────────────────────────────
        with h5py.File(str(h5_path), "r") as f:
            images_np = f["images"][:]   # (4, 160, 208, 160) float32

        images_t = torch.from_numpy(images_np).unsqueeze(0).to(device)  # (1,4,160,208,160)

        # ── 3. Inference ──────────────────────────────────────────────────────
        if args.mc_passes > 0:
            mean_pred, _ = mc_inference(model, images_t, n_passes=args.mc_passes)
            pred_crop = mean_pred.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        else:
            with torch.no_grad():
                pred_crop = model(images_t).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        # pred_crop: (160, 208, 160)  labels {0, 1, 2, 3}

        # ── 4. Reverse center_pad ─────────────────────────────────────────────
        pred_brain = reverse_center_pad(pred_crop, brain_shape)

        # ── 5. Place into original (182, 218, 182) space ──────────────────────
        pred_full = np.zeros(orig_shape, dtype=np.uint8)
        pred_full[x0:x1, y0:y1, z0:z1] = pred_brain

        # ── 6. Save NIfTI ─────────────────────────────────────────────────────
        out_img = nib.Nifti1Image(pred_full, affine, header)
        out_img.set_data_dtype(np.uint8)
        nib.save(out_img, str(out_nii))

    # ── Zip all predictions ───────────────────────────────────────────────────────
    zip_path = Path("predictions.zip")
    pred_files = sorted(OUT_DIR.glob("*-seg.nii.gz"))
    print(f"\nPacking {len(pred_files)} predictions → {zip_path}")
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for p in pred_files:
            zf.write(str(p), arcname=p.name)

    print(f"\nDone.")
    print(f"  Predictions : {OUT_DIR}/  ({len(pred_files)} files)")
    print(f"  Upload file : {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
    print(f"\nSubmission steps:")
    print(f"  1. Go to https://www.synapse.org  and log in")
    print(f"  2. Navigate to the BraTS 2024 GLI challenge page")
    print(f"  3. Upload predictions.zip to the submission portal")


if __name__ == "__main__":
    main()