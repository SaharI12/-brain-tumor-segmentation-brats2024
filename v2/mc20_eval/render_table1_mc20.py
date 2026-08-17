"""Render Table 1 of the paper: v1 vs v2.1 under MC Dropout (n = 20 passes).

Supersedes render_mc20_table.py: the Mean column is now a plain unweighted
average of the three region means with no ±. The previous version printed the
average-of-region-means next to the std of a *different* estimator (the
per-subject region average), so the centre and the spread in that cell
described different quantities.

Medians are not shown in the table (HD95 is heavy-tailed — std ~9 mm, max
~95 mm — so they are quoted in the text instead); they are still printed to
stdout for reference.

Reads the per-subject arrays written by mc20_compare.py, in this directory.
Output: v2/results_summary/table1_v1_vs_v2_mc20.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results_summary/table1_v1_vs_v2_mc20.png"

LABELS = {"v1": "3D-UNet", "v2_1": "3D-UNet + Attention Gate"}
COLS = ["Dice TC", "Dice WT", "Dice ET", "Dice Mean",
        "HD95 TC (mm)", "HD95 WT (mm)", "HD95 ET (mm)", "HD95 Mean (mm)"]

LABEL_FONTSIZE = 13
VALUE_FONTSIZE = 13


def wrap_label(label, threshold=9):
    """Wrap onto two lines at the space closest to the middle."""
    if len(label) <= threshold or " " not in label:
        return label
    spaces = [i for i, ch in enumerate(label) if ch == " "]
    best = min(spaces, key=lambda i: abs(i - len(label) / 2))
    return label[:best] + "\n" + label[best + 1:]


summary = json.load(open(HERE / "mc20_summary.json"))
data, rows, mean_dice = {}, {}, {}

for key, label in LABELS.items():
    dice = np.load(HERE / f"mc20_{key}_dice.npy")   # (n_subjects, 3) — TC / WT / ET
    hd = np.load(HERE / f"mc20_{key}_hd95.npy")
    data[key] = (dice, hd)

    cells = []
    for r in range(3):
        col = dice[:, r]
        cells.append(f"{np.nanmean(col):.4f} ± {np.nanstd(col):.4f}")
    cells.append(f"{np.nanmean(dice, axis=0).mean():.4f}")
    for r in range(3):
        col = hd[:, r]
        cells.append(f"{np.nanmean(col):.2f} ± {np.nanstd(col):.2f}")
    cells.append(f"{np.nanmean(hd, axis=0).mean():.2f}")
    rows[label] = cells
    mean_dice[label] = float(np.nanmean(dice, axis=0).mean())

better = max(mean_dice, key=mean_dice.get)

n_scored = summary["v2_1"]["n_scored"]
n_total = summary["v2_1"]["n_subjects"]

title = ("BraTS 2024 GLI — Dice / HD95 with MC Dropout (n = 20 passes): "
         "3D-UNet vs 3D-UNet + Attention Gate")
footnote = (
    f"Mean ± standard deviation across the {n_total} held-out test subjects; "
    "both models share the identical split (seed 42) and neither saw these subjects in training.\n"
    "MC Dropout inference — 20 stochastic passes, softmax averaged, argmax of the mean. Dropout p = 0.2 (3D-UNet) / 0.15 (+ Attention Gate), "
    "same placement in both (bottleneck + two deepest decoder blocks).\n"
    f"A region absent from the ground truth is excluded from that region's statistics (scored n = {n_scored[0]} TC / {n_scored[1]} WT / {n_scored[2]} ET). "
    "Mean = unweighted average of the three region means."
)

model_labels = [LABELS["v1"], LABELS["v2_1"]]
wrapped_cols = [wrap_label(c) for c in COLS]
wrapped_models = [wrap_label(m) for m in model_labels]

TITLE_H, FOOTNOTE_LINE_H, FOOTNOTE_PAD, TOP_MARGIN = 0.55, 0.16, 0.18, 0.15
n_cols, n_rows = len(COLS), len(model_labels)
n_footnote_lines = footnote.count("\n") + 1
header_lines = max(c.count("\n") + 1 for c in wrapped_cols)
cell_lines = max(str(v).count("\n") + 1 for row in rows.values() for v in row)
row_label_lines = max(m.count("\n") + 1 for m in wrapped_models)

HEADER_H = 0.18 + 0.30 * header_lines
ROW_H = 0.20 + 0.30 * max(cell_lines, row_label_lines)
table_block_h = HEADER_H + ROW_H * n_rows
footnote_block_h = FOOTNOTE_LINE_H * n_footnote_lines + FOOTNOTE_PAD

widest_cell = max(len(line) for row in rows.values() for v in row for line in str(v).split("\n"))
col_w = max(1.4, 0.105 * widest_cell + 0.35)
fig_w = col_w * n_cols + 0.09 * max(len(m) for m in model_labels) + 1.6
fig_h = TOP_MARGIN + TITLE_H + table_block_h + footnote_block_h

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("white")
ax.axis("off")

bbox_bottom = footnote_block_h / fig_h
bbox_top = (footnote_block_h + table_block_h) / fig_h

tbl = ax.table(
    cellText=[rows[m] for m in model_labels], rowLabels=wrapped_models,
    colLabels=wrapped_cols, cellLoc="center",
    bbox=[0, bbox_bottom, 1, bbox_top - bbox_bottom],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(VALUE_FONTSIZE)
tbl.auto_set_column_width(col=list(range(-1, n_cols)))

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    cell.set_facecolor("white")
    cell.set_text_props(color="black")
    if c == -1:                                  # model names
        cell.set_fontsize(LABEL_FONTSIZE)
        cell.set_text_props(ha="left", color="black", fontweight="bold")
        cell.set_facecolor("#e8e8e8" if r > 0 else "white")
        cell.PAD = 0.04
    elif r == 0:                                 # metric names
        cell.set_fontsize(LABEL_FONTSIZE)
        cell.set_facecolor("#e8e8e8")
        cell.set_text_props(ha="center", color="black", fontweight="bold")
        cell.PAD = 0.04
    elif model_labels[r - 1] == better:
        cell.set_text_props(color="black", fontweight="bold")

ax.set_title(title, color="black", fontsize=11, fontweight="bold", pad=14)
fig.text(0.5, 0.01, footnote, ha="center", va="bottom", color="#555555", fontsize=7.5)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(OUT), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {OUT}")
for key, label in LABELS.items():
    dice, hd = data[key]
    med_d = "/".join(f"{np.nanmedian(dice[:, r]):.3f}" for r in range(3))
    med_h = "/".join(f"{np.nanmedian(hd[:, r]):.2f}" for r in range(3))
    print(f"  {label}: mean Dice {mean_dice[label]:.4f}  "
          f"median Dice TC/WT/ET {med_d}  median HD95 TC/WT/ET {med_h} mm")
