"""Re-run the v1-vs-v2.1 comparison with MC Dropout (n=20) on the shared 324-subject split.

Both models are evaluated with dropout active at inference: 20 stochastic passes,
softmax averaged, argmax of the mean = final segmentation. Reports per-subject
Dice / HD95 so mean +/- std (across subjects) can be quoted.

Dice convention matches evaluate_v2.py / the existing comparison table:
a region absent from the ground truth is excluded (NaN), not scored 1.0.
"""
import argparse, json, sys, time
from pathlib import Path

import h5py, numpy as np, torch
from monai.metrics import HausdorffDistanceMetric

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "unet3d_baseline"))   # v1: model.UNet3D
sys.path.insert(0, str(ROOT / "v2"))                # v2.1: model_v2.UNet3DAttn

DATA = ROOT / "processed/train"
SPLIT = ROOT / "v2/checkpoints_v2_1/val_split.json"
# Results land next to this script unless --out_dir says otherwise.
DEFAULT_OUT = Path(__file__).resolve().parent

REGIONS = ("TC", "WT", "ET")


def region_masks(arr):
    """arr: int tensor/array -> (TC, WT, ET) boolean masks."""
    return ((arr == 1) | (arr == 3)), (arr > 0), (arr == 3)


def build_model(which, device):
    if which == "v1":
        from model import UNet3D
        ck = torch.load(ROOT / "unet3d_baseline/checkpoints/best_model.pth", map_location=device)
        a = ck.get("args", {})
        m = UNet3D(4, 4, init_features=a.get("init_features", 32),
                   dropout_p=a.get("dropout", 0.2)).to(device)
    else:
        from model_v2 import UNet3DAttn
        ck = torch.load(ROOT / "v2/checkpoints_v2_1/best_model.pth", map_location=device)
        a = ck.get("args", {})
        m = UNet3DAttn(4, 4, init_features=a.get("init_features", 32),
                       dropout_p=a.get("dropout", 0.15)).to(device)
    m.load_state_dict(ck["model_state_dict"])
    m.train()          # dropout stays ACTIVE for MC inference
    print(f"  {which}: epoch {ck.get('epoch')}  best_dice {ck.get('best_dice'):.4f}  "
          f"dropout_p {a.get('dropout')}", flush=True)
    return m


def evaluate(which, files, n_passes, device):
    model = build_model(which, device)
    hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95,
                                        reduction="mean_batch")
    dice_rows, hd_rows = [], []
    t0 = time.time()
    for i, fn in enumerate(files, 1):
        with h5py.File(DATA / fn, "r") as f:
            img, gt = f["images"][:], f["seg"][:]
        x = torch.from_numpy(img).unsqueeze(0).to(device)

        prob_sum = None
        with torch.no_grad():
            for _ in range(n_passes):
                p = torch.softmax(model(x), dim=1)
                prob_sum = p if prob_sum is None else prob_sum + p
        pred = (prob_sum / n_passes).argmax(1)[0].cpu().numpy()
        del prob_sum

        d, h = [], []
        pm, gm = region_masks(pred), region_masks(gt)
        for r in range(3):
            p_r, g_r = pm[r], gm[r]
            if not g_r.any():                    # empty GT region -> excluded
                d.append(np.nan); h.append(np.nan); continue
            denom = p_r.sum() + g_r.sum()
            d.append(2.0 * np.logical_and(p_r, g_r).sum() / denom)
            hd_metric.reset()
            hd_metric(y_pred=torch.from_numpy(p_r).float()[None, None],
                      y=torch.from_numpy(g_r).float()[None, None])
            v = float(hd_metric.aggregate().item())
            h.append(v if np.isfinite(v) else np.nan)
        dice_rows.append(d); hd_rows.append(h)

        if i % 25 == 0:
            el = time.time() - t0
            print(f"  [{which}] {i}/{len(files)}  {el/i:.1f}s/subj  "
                  f"eta {(len(files)-i)*el/i/60:.0f}min", flush=True)
    del model
    torch.cuda.empty_cache()
    return np.array(dice_rows), np.array(hd_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--tag", default="mc20")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if cached .npy results exist")
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT),
                    help="Where .npy / summary.json are read and written")
    args = ap.parse_args()

    global OUT
    OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    device = torch.device("cuda")
    files = json.load(open(SPLIT))["val"]
    if args.limit:
        files = files[:args.limit]
    print(f"{len(files)} subjects, {args.passes} MC passes each", flush=True)

    out = {}
    for which in ("v1", "v2_1"):
        dice_path = OUT / f"{args.tag}_{which}_dice.npy"
        hd_path = OUT / f"{args.tag}_{which}_hd95.npy"
        if dice_path.exists() and hd_path.exists() and not args.force:
            # Already computed in an earlier (possibly interrupted) run — ~1h of GPU each.
            dice, hd = np.load(dice_path), np.load(hd_path)
            print(f"  [{which}] reusing cached results ({len(dice)} subjects) "
                  f"from {dice_path.name}", flush=True)
        else:
            dice, hd = evaluate(which, files, args.passes, device)
        out[which] = {
            "n_subjects": len(files),
            "dice_mean": np.nanmean(dice, 0).tolist(),
            "dice_std":  np.nanstd(dice, 0).tolist(),
            "hd95_mean": np.nanmean(hd, 0).tolist(),
            "hd95_std":  np.nanstd(hd, 0).tolist(),
            "n_scored":  (~np.isnan(dice)).sum(0).tolist(),
        }
        np.save(dice_path, dice)
        np.save(hd_path, hd)
        m, s = out[which]["dice_mean"], out[which]["dice_std"]
        hm, hs = out[which]["hd95_mean"], out[which]["hd95_std"]
        print(f"\n{which}  Dice " + "  ".join(f"{r} {m[i]:.4f}+-{s[i]:.4f}"
                                              for i, r in enumerate(REGIONS)))
        print(f"{which}  HD95 " + "  ".join(f"{r} {hm[i]:.2f}+-{hs[i]:.2f}"
                                            for i, r in enumerate(REGIONS)), flush=True)

    with open(OUT / f"{args.tag}_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUT}/{args.tag}_summary.json")


if __name__ == "__main__":
    main()
