import os
import random
from glob import glob

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class BraTSDataset(Dataset):
    def __init__(self, source, augment=False):
        """
        Args:
            source: path to directory of .h5 files, or explicit list of file paths
            augment: if True, applies random 90-degree 3D rotations
        """
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
            images = f["images"][:]        # (4, 160, 208, 160) float32
            seg    = f["seg"][:].astype(np.int64)  # (160, 208, 160)

        images = torch.from_numpy(images)
        seg    = torch.from_numpy(seg)

        if self.augment:
            images, seg = self._random_rotate(images, seg)

        return images, seg

    def _random_rotate(self, images, seg):
        # Rotate in H-D plane only: H=D=160, so all k values are shape-preserving.
        # W=208 is different so we skip H-W and W-D rotations.
        k = random.randint(0, 3)
        if k:
            images = torch.rot90(images, k, dims=[1, 3])
            seg    = torch.rot90(seg,    k, dims=[0, 2])
        return images, seg


def get_split_loaders(train_dir, val_split=0.2, batch_size=1, num_workers=4, seed=42):
    """Split processed/train into train/val by ratio. Returns (train_loader, val_loader)."""
    all_files = sorted(glob(os.path.join(train_dir, "*.h5")))
    rng = random.Random(seed)
    rng.shuffle(all_files)
    n_val = int(len(all_files) * val_split)
    val_files   = all_files[:n_val]
    train_files = all_files[n_val:]

    train_loader = DataLoader(
        BraTSDataset(train_files, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        BraTSDataset(val_files, augment=False),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    ds = BraTSDataset("../processed/train", augment=True)
    print(f"Train subjects: {len(ds)}")
    images, seg = ds[0]
    print(f"Images: {images.shape}  dtype={images.dtype}  min={images.min():.4f}  max={images.max():.4f}")
    print(f"Seg:    {seg.shape}  dtype={seg.dtype}  labels={seg.unique().tolist()}")

    train_loader, val_loader = get_split_loaders("../processed/train")
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")
    imgs, segs = next(iter(train_loader))
    print(f"Batch images: {imgs.shape}  Batch seg: {segs.shape}")