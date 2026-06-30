import json
import os
import random
from glob import glob

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class BraTSDatasetV2(Dataset):
    """BraTS dataset with a stronger augmentation pipeline (train only).

    Augmentations applied when augment=True:
      - Random flip along each of the 3 spatial axes independently (p=0.5 each)
      - Per-modality intensity scale  ~ U[0.85, 1.15]  (p=0.5 per modality)
      - Per-modality intensity shift  ~ U[-0.10, 0.10] (p=0.5 per modality)
      - Gaussian noise N(0, 0.01) added to all modalities            (p=0.20)

    Validation instances are NEVER augmented — pass augment=False (default).
    """

    def __init__(self, source, augment=False):
        if isinstance(source, list):
            self.files = source
        else:
            self.files = sorted(glob(os.path.join(source, "*.h5")))
        if not self.files:
            raise RuntimeError(f"No .h5 files found in {source}")
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with h5py.File(self.files[idx], "r") as f:
            images = f["images"][:].astype(np.float32)  # (4, H, W, D)
            seg    = f["seg"][:].astype(np.int64)        # (H, W, D)

        if self.augment:
            images, seg = self._augment(images, seg)

        return torch.from_numpy(images), torch.from_numpy(seg)

    def _augment(self, images, seg):
        # Random flips along H, W, D axes
        for axis in range(3):
            if random.random() < 0.5:
                images = np.flip(images, axis=axis + 1)  # +1 skips channel dim
                seg    = np.flip(seg,    axis=axis)

        # Per-modality intensity scale and shift — each modality independently
        for c in range(images.shape[0]):
            if random.random() < 0.5:
                images[c] *= random.uniform(0.85, 1.15)
            if random.random() < 0.5:
                images[c] += random.uniform(-0.10, 0.10)
        images = np.clip(images, 0.0, 1.0)

        # Additive Gaussian noise
        if random.random() < 0.20:
            noise  = np.random.normal(0.0, 0.01, images.shape).astype(np.float32)
            images = np.clip(images + noise, 0.0, 1.0)

        # np.flip returns a view with negative strides — make contiguous for torch
        images = np.ascontiguousarray(images)
        seg    = np.ascontiguousarray(seg)

        return images, seg


def get_split_loaders_v2(
    train_dir,
    val_split=0.2,
    batch_size=1,
    num_workers=4,
    seed=42,
    split_save_path=None,
):
    """Split processed/train into train/val DataLoaders with a persistent val split.

    Clean validation guarantee:
    - On the first run, the split is saved as JSON (filenames only, no full paths)
      to split_save_path so the same subjects are always in the val set.
    - On resume (split_save_path already exists), the saved list is reloaded —
      subject assignment never shifts due to file additions or seed changes.
    - Val DataLoader always uses augment=False.

    Args:
        train_dir:       directory of processed .h5 files
        val_split:       fraction for validation (ignored if split already saved)
        batch_size:      training batch size (val is always 1)
        num_workers:     DataLoader workers
        seed:            RNG seed for the initial shuffle (ignored on resume)
        split_save_path: path to JSON file for persistent split; None = no save/load

    Returns:
        (train_loader, val_loader)
    """
    all_files = sorted(glob(os.path.join(train_dir, "*.h5")))
    if not all_files:
        raise RuntimeError(f"No .h5 files in {train_dir}")

    if split_save_path and os.path.exists(split_save_path):
        with open(split_save_path) as f:
            saved = json.load(f)
        val_files   = [os.path.join(train_dir, fn) for fn in saved["val"]]
        train_files = [os.path.join(train_dir, fn) for fn in saved["train"]]
        print(f"Val split loaded from {split_save_path}  "
              f"(train {len(train_files)}, val {len(val_files)})")
    else:
        rng = random.Random(seed)
        shuffled = list(all_files)
        rng.shuffle(shuffled)
        n_val       = int(len(shuffled) * val_split)
        val_files   = shuffled[:n_val]
        train_files = shuffled[n_val:]

        if split_save_path:
            os.makedirs(os.path.dirname(os.path.abspath(split_save_path)), exist_ok=True)
            with open(split_save_path, "w") as fh:
                json.dump(
                    {
                        "seed":      seed,
                        "val_split": val_split,
                        "n_total":   len(all_files),
                        "val":       [os.path.basename(p) for p in val_files],
                        "train":     [os.path.basename(p) for p in train_files],
                    },
                    fh,
                    indent=2,
                )
            print(f"Val split saved to {split_save_path}  "
                  f"(train {len(train_files)}, val {len(val_files)})")

    train_loader = DataLoader(
        BraTSDatasetV2(train_files, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        BraTSDatasetV2(val_files, augment=False),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    ds = BraTSDatasetV2("processed/train", augment=True)
    print(f"Subjects: {len(ds)}")
    images, seg = ds[0]
    print(f"Images: {images.shape}  dtype={images.dtype}  [{images.min():.3f}, {images.max():.3f}]")
    print(f"Seg:    {seg.shape}  dtype={seg.dtype}  labels={seg.unique().tolist()}")