"""
Assemble the final MC-dropout / Dice-HD95 comparison, all evaluated on the SAME
324-subject BraTS 2024 persistent val split:

  - v2.1      — our Attention U-Net (from evaluate_v2.py --json_out)
  - SegResNet — MONAI's published BraTS 2018 baseline (from run_baseline_segresnet.py)
  - MedNeXt   — Ferreira et al.'s published BraTS 2024 baseline, dropout added by us
                (from run_baseline_mednext.py)

Outputs to v2/baseline_comparison/:
  - comparison_table.png    : Dice/HD95, TC/WT/ET/Mean, det + MC-20, all models
  - entropy_comparison.png  : whole-tumor-region entropy at TP/FP/FN, ours vs. baselines
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Fallback v2.1 numbers (from PROGRESS.md), used only if --v21_det_json / --v21_mc_json
# are not found, e.g. before evaluate_v2.py --json_out has been (re-)run.
V21_FALLBACK = {
    "dice_det": {"TC": 0.8791, "WT": 0.9206, "ET": 0.8613, "mean": 0.8870},
    "hd95_det": {"TC": 4.83, "WT": 4.87, "ET": 5.06, "mean": 4.92},
    "dice_mc":  {"TC": 0.8794, "WT": 0.9205, "ET": 0.8616, "mean": 0.8872},
    "hd95_mc":  {"TC": 4.81, "WT": 4.88, "ET": 5.04, "mean": 4.91},
}

REGIONS = ["TC", "WT", "ET"]
BG = "black"


def load_json(path):
    if path and Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


def build_comparison_table(models, out_path: Path):
    """models: list of (label, results_dict) pairs, in display order."""
    rows = []

    def add_row(label, d):
        if d is None:
            return
        for key, dice_d, hd_d in [
            ("Deterministic", d.get("dice_det"), d.get("hd95_det")),
            ("MC Dropout (20)", d.get("dice_mc"), d.get("hd95_mc")),
        ]:
            if dice_d is None:
                continue
            rows.append([f"{label} — {key}"] + [f"{dice_d[r]:.4f}" for r in REGIONS] +
                        [f"{dice_d['mean']:.4f}"] + [f"{hd_d[r]:.2f}" for r in REGIONS] +
                        [f"{hd_d['mean']:.2f}"])

    for label, d in models:
        add_row(label, d)

    col_labels = ["Model — Mode", "Dice TC", "Dice WT", "Dice ET", "Dice Mean",
                  "HD95 TC", "HD95 WT", "HD95 ET", "HD95 Mean"]

    col_widths = [0.30] + [0.0875] * (len(col_labels) - 1)

    fig, ax = plt.subplots(figsize=(17, 0.55 * len(rows) + 1.2))
    fig.patch.set_facecolor(BG)
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center",
                   colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#555")
        if c == 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.02
        if r == 0:
            cell.set_facecolor("#262626")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(BG)
            cell.set_text_props(color="white")
    ax.set_title("BraTS 2024 GLI — Dice / HD95 Comparison\n"
                  "(all models evaluated on the same 324-subject val split — see BASELINE_COMPARISON.md)",
                  color="white", fontsize=10, fontweight="bold", pad=12)
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


def build_entropy_comparison(v21_uncertainty, baselines, out_path: Path):
    """
    v2.1's own entropy stats are whole-tumor binary (single TN/TP/FP/FN set).
    Baselines' are per-region (TC/WT/ET). We compare against each baseline's
    WT-channel entropy since WT is the closest analog to v2.1's whole-tumor grouping.

    baselines: list of (label, results_dict, color) triples.
    """
    if v21_uncertainty is None:
        print("  Skipping entropy comparison — missing v2.1 uncertainty summary.")
        return
    baselines = [(label, d, color) for label, d, color in baselines
                 if d is not None and "entropy_by_region" in d]

    regions_order = ["TP", "FP", "FN"]
    series = [("Ours (v2.1, WT binary)",
               [v21_uncertainty["entropy_by_region_mean"][r] for r in regions_order], "#3dcc9a")]
    for label, d, color in baselines:
        wt = d["entropy_by_region"]["WT"]
        series.append((label, [wt.get(r, {}).get("mean", np.nan) for r in regions_order], color))

    n_series = len(series)
    x = np.arange(len(regions_order))
    width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for i, (label, means, color) in enumerate(series):
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(x + offset, means, width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(regions_order, color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="white", labelsize=9)
    ax.set_ylabel("Mean Predictive Entropy", color="white", fontsize=10)
    ax.set_title("MC Dropout Uncertainty — Ours vs. Published Baselines\n(Whole Tumor region)",
                 color="white", fontsize=11, fontweight="bold")
    for sp in ax.spines.values():
        sp.set_color("#555")
    ax.grid(axis="y", color="#333", linewidth=0.7)
    ax.legend(fontsize=8.5, labelcolor="white", facecolor=BG, edgecolor="#555")

    plt.tight_layout(pad=1.5)
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Compare v2.1 vs. published baselines")
    p.add_argument("--v21_det_json", default="baseline_comparison/v21_det.json")
    p.add_argument("--v21_mc_json", default="baseline_comparison/v21_mc20.json")
    p.add_argument("--segresnet_json", default="baseline_comparison/segresnet_results.json")
    p.add_argument("--mednext_json", default="baseline_comparison/mednext_results.json")
    p.add_argument("--v21_uncertainty_json", default="uncertainty_vis_v2_1/summary.json")
    p.add_argument("--out_dir", default="baseline_comparison")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v21_det = load_json(args.v21_det_json)
    v21_mc  = load_json(args.v21_mc_json)
    if v21_det is None and v21_mc is None:
        print(f"  No v2.1 --json_out results found — falling back to PROGRESS.md numbers")
        v21 = dict(V21_FALLBACK)
    else:
        v21 = {}
        if v21_det:
            v21["dice_det"] = v21_det["dice_det"]
            v21["hd95_det"] = v21_det["hd95_det"]
        if v21_mc:
            v21["dice_mc"] = v21_mc["dice_mc"]
            v21["hd95_mc"] = v21_mc["hd95_mc"]

    segresnet = load_json(args.segresnet_json)
    if segresnet is None:
        raise FileNotFoundError(
            f"{args.segresnet_json} not found — run run_baseline_segresnet.py first."
        )
    mednext = load_json(args.mednext_json)
    if mednext is None:
        print(f"  No MedNeXt results found at {args.mednext_json} — omitting from comparison.")

    v21_uncertainty = load_json(args.v21_uncertainty_json)

    print("Building comparison table...")
    models = [
        ("v2.1 (ours, Attention U-Net)", v21),
        ("SegResNet (published, BraTS 2018)", segresnet),
        ("MedNeXt (published, BraTS 2024, +our dropout)", mednext),
    ]
    build_comparison_table(models, out_dir / "comparison_table.png")

    print("Building entropy comparison figure...")
    baselines = [
        ("SegResNet baseline (WT channel)", segresnet, "#ff7320"),
        ("MedNeXt baseline (WT channel)", mednext, "#4477ff"),
    ]
    build_entropy_comparison(v21_uncertainty, baselines, out_dir / "entropy_comparison.png")

    print(f"\nDone. Output in {out_dir}/")


if __name__ == "__main__":
    main()
