import argparse
import os
import random
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric

from dataset import get_split_loaders
from model import UNet3D, get_region_masks


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate(model, val_loader, dice_metric, device):
    model.eval()
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="             [val]  ", leave=False)
        for images, seg in val_bar:
            images, seg = images.to(device), seg.to(device)
            logits      = model(images)
            pred_class  = logits.argmax(dim=1)

            TC_pred, WT_pred, ET_pred = get_region_masks(pred_class)
            TC_gt,   WT_gt,   ET_gt   = get_region_masks(seg)

            pred_r = torch.stack([TC_pred, WT_pred, ET_pred], dim=1).float()
            gt_r   = torch.stack([TC_gt,   WT_gt,   ET_gt],   dim=1).float()
            dice_metric(y_pred=pred_r, y=gt_r)

    scores = dice_metric.aggregate().cpu().numpy()  # (3,) — TC, WT, ET
    dice_metric.reset()
    return scores


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler,
                    best_mean_dice, no_improve, dice_scores, args):
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "best_dice":            best_mean_dice,
            "no_improve":           no_improve,
            "val_dice_TC_WT_ET":    dice_scores.tolist(),
            "args":                 vars(args),
        },
        path,
    )


def train(args):
    set_seed(args.seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  AMP: {use_amp}")

    train_loader, val_loader = get_split_loaders(
        train_dir=args.data_dir,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"Train: {len(train_loader.dataset)} subjects  Val: {len(val_loader.dataset)} subjects")

    model = UNet3D(
        in_channels=4,
        out_channels=4,
        init_features=args.init_features,
        dropout_p=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    loss_fn     = DiceFocalLoss(include_background=False, to_onehot_y=True, softmax=True, gamma=2.0)
    optimizer   = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler   = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler      = torch.amp.GradScaler("cuda", enabled=use_amp)
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=True)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_mean_dice = -1.0
    no_improve     = 0
    start_epoch    = 1

    # ── Resume ────────────────────────────────────────────────────────────────
    latest_path = os.path.join(args.ckpt_dir, "latest_checkpoint.pth")
    if args.resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_mean_dice = ckpt["best_dice"]
        no_improve     = ckpt["no_improve"]
        start_epoch    = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}  best Dice {best_mean_dice:.4f}  no_improve {no_improve}")
    elif args.resume:
        print(f"No checkpoint found at {latest_path} — starting from scratch")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs} [train]", leave=False)
        for step, (images, seg) in enumerate(train_bar, 1):
            images = images.to(device)
            seg    = seg.to(device).unsqueeze(1)   # (B, 1, H, W, D) — required by to_onehot_y

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss   = loss_fn(logits, seg)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

            if args.debug and step >= 5:
                break

        scheduler.step()
        avg_loss = epoch_loss / step

        dice_scores = validate(model, val_loader, dice_metric, device)
        mean_dice   = float(dice_scores.mean())

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss {avg_loss:.4f} | "
            f"Dice  TC {dice_scores[0]:.4f}  WT {dice_scores[1]:.4f}  ET {dice_scores[2]:.4f}  "
            f"Mean {mean_dice:.4f} | "
            f"LR {scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.0f}s"
        )

        # ── Save latest every epoch (for resume) ──────────────────────────────
        save_checkpoint(
            latest_path, epoch, model, optimizer, scheduler, scaler,
            best_mean_dice, no_improve, dice_scores, args,
        )

        # ── Save best + early stopping ─────────────────────────────────────────
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            no_improve     = 0
            save_checkpoint(
                os.path.join(args.ckpt_dir, "best_model.pth"),
                epoch, model, optimizer, scheduler, scaler,
                best_mean_dice, no_improve, dice_scores, args,
            )
            print(f"  → Saved best model  (mean Dice {best_mean_dice:.4f})")
        else:
            no_improve += 1
            print(f"  → No improvement ({no_improve}/{args.patience})")
            if no_improve >= args.patience:
                print(f"Early stopping triggered after {no_improve} epochs without improvement.")
                break

        if args.debug:
            break


def parse_args():
    p = argparse.ArgumentParser(description="BraTS 2024 3D U-Net Training")
    p.add_argument("--data_dir",      default="/storage/sahar/CV_project/data/train")
    p.add_argument("--ckpt_dir",      default="/storage/sahar/CV_project/checkpoints")
    p.add_argument("--init_features", type=int,   default=32,   help="16 locally, 32 on RunAI")
    p.add_argument("--epochs",        type=int,   default=300)
    p.add_argument("--batch_size",    type=int,   default=1)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--dropout",       type=float, default=0.2)
    p.add_argument("--val_split",     type=float, default=0.2)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--patience",      type=int,   default=20,   help="Early stopping patience (epochs)")
    p.add_argument("--resume",        action="store_true",      help="Resume from latest_checkpoint.pth in ckpt_dir")
    p.add_argument("--debug",         action="store_true",      help="5 steps, 1 epoch smoke test")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
