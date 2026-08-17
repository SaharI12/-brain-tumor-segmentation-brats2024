"""
v1-vs-v2 comparison tables (models as rows, metrics as columns) as PNGs.

Numbers are hardcoded below from evaluate_v2.py's runs (results_summary/v21_det.json)
and the uncertainty analysis — this script just renders them as white-background /
black-text table images for slides/reports.

Outputs:
  results_summary/v1_vs_v2_results_table.png   — Dice / HD95
  results_summary/v1_vs_v2_entropy_table.png   — entropy / ECE / AUROC
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent / "results_summary"
NOT_COMPUTED = "not computed"

MODEL_LABELS = ["3D-UNet", "3D-UNet + Attention Gate"]

RESULTS_COLS = ["Dice TC", "Dice WT", "Dice ET", "Dice Mean",
                "HD95 TC (mm)", "HD95 WT (mm)", "HD95 ET (mm)", "HD95 Mean (mm)"]
RESULTS_ROWS = {
    "3D-UNet":                  ["0.8638", "0.9087", "0.8503", "0.8743", "5.65", "6.11", "5.74", "5.83"],
    "3D-UNet + Attention Gate": ["0.8791", "0.9206", "0.8613", "0.8870", "4.83", "4.87", "5.06", "4.92"],
}
# Higher Dice is better, lower HD95 is better — the attention-gate model wins every column here.
RESULTS_BETTER = "3D-UNet + Attention Gate"
RESULTS_TITLE = "BraTS 2024 GLI — Dice / HD95: 3D-UNet vs 3D-UNet + Attention Gate"
RESULTS_FOOTNOTE = ("Eval set — 3D-UNet: own 80/20 holdout, reshuffled split.  "
                     "3D-UNet + Attention Gate: persistent 324-subject val split.")

ENTROPY_COLS = ["Entropy TN", "Entropy TP", "Entropy FP", "Entropy FN",
                "FP/TP Ratio", "ECE", "AUROC"]
ENTROPY_ROWS = {
    "3D-UNet":                  [NOT_COMPUTED, "0.039", "0.236", "~0.20", "~6.0×", NOT_COMPUTED, NOT_COMPUTED],
    "3D-UNet + Attention Gate": ["0.0002", "0.0285", "0.1960", "0.1112", "~6.9×", "0.0080", "0.8956"],
}
ENTROPY_TITLE = "BraTS 2024 GLI — MC Dropout Entropy / Calibration: 3D-UNet vs 3D-UNet + Attention Gate"
ENTROPY_FOOTNOTE = ("Directionally comparable, not a controlled rerun — 3D-UNet numbers are from an\n"
                     "unregenerated script run on a different (reshuffled) holdout; TN/ECE/AUROC weren't\n"
                     "computed for 3D-UNet since its uncertainty script predates the calibration analysis.")

# Single-model entropy-only view (no 3D-UNet row)
V2_MODEL_LABEL = "3D-UNet + Attention Gate"
ENTROPY_ONLY_COLS = ["Entropy TN", "Entropy TP", "Entropy FP", "Entropy FN", "FP/TP Ratio", "ECE", "AUROC"]
ENTROPY_ONLY_ROWS = {
    V2_MODEL_LABEL: ["0.0002", "0.0285", "0.1960", "0.1112", "~6.9×", "0.0080", "0.8956"],
}
ENTROPY_ONLY_TITLE = "BraTS 2024 GLI — MC Dropout Entropy / Calibration: 3D-UNet + Attention Gate"
ENTROPY_ONLY_FOOTNOTE = ("Predictive entropy H = −Σ p̄_c log(p̄_c), mean softmax over 20 MC Dropout passes.\n"
                          "TN/TP/FP/FN bucketed by whole-tumor binary mask (prediction vs. ground truth).\n"
                          "ECE = mean |accuracy − confidence| over 15 confidence bins; AUROC = entropy's ROC-AUC\n"
                          "for detecting voxel-level errors (both: brain-mask voxels, same 50-subject sample).")


LABEL_FONTSIZE = 14
VALUE_FONTSIZE = 14


def wrap_label(label: str, threshold: int = 9) -> str:
    """Wrap onto two lines at the space closest to the middle, for balanced headers/row labels."""
    if len(label) <= threshold or " " not in label:
        return label
    space_positions = [i for i, ch in enumerate(label) if ch == " "]
    mid = len(label) / 2
    best = min(space_positions, key=lambda i: abs(i - mid))
    return label[:best] + "\n" + label[best + 1:]


def build_table(model_labels, col_labels, row_data, better_model, title, footnote, out_path: Path):
    n_cols = len(col_labels)
    wrapped_col_labels = [wrap_label(label) for label in col_labels]
    wrapped_model_labels = [wrap_label(label) for label in model_labels]

    # Layout is stacked title / table / footnote, sized in inches so the footnote block
    # (and the header/row-label rows, which can wrap to 2 lines) grow with their actual
    # line count instead of a fixed axes fraction that stops fitting them.
    TITLE_H, FOOTNOTE_LINE_H, FOOTNOTE_PAD, TOP_MARGIN = 0.55, 0.14, 0.16, 0.15

    n_rows = len(model_labels)
    n_footnote_lines = footnote.count("\n") + 1
    header_lines = max(label.count("\n") + 1 for label in wrapped_col_labels)
    row_label_lines = max(label.count("\n") + 1 for label in wrapped_model_labels)
    HEADER_H = 0.18 + 0.30 * header_lines
    ROW_H = 0.16 + 0.28 * row_label_lines
    table_block_h = HEADER_H + ROW_H * n_rows
    footnote_block_h = FOOTNOTE_LINE_H * n_footnote_lines + FOOTNOTE_PAD

    row_label_w = max(len(m) for m in model_labels)
    # Column width follows the widest cell so long values ("0.8789 ± 0.1417") get the
    # room they need; short-valued tables stay at the original 1.4in per column.
    widest_cell = max(len(str(v)) for row in row_data.values() for v in row)
    col_w = max(1.4, 0.105 * widest_cell + 0.35)
    fig_w = col_w * n_cols + 0.09 * row_label_w + 1.6
    fig_h = TOP_MARGIN + TITLE_H + table_block_h + footnote_block_h
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    bbox_bottom = footnote_block_h / fig_h
    bbox_top = (footnote_block_h + table_block_h) / fig_h

    cell_text = [row_data[m] for m in model_labels]
    tbl = ax.table(
        cellText=cell_text, rowLabels=wrapped_model_labels, colLabels=wrapped_col_labels, cellLoc="center",
        bbox=[0, bbox_bottom, 1, bbox_top - bbox_bottom],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(VALUE_FONTSIZE)
    tbl.auto_set_column_width(col=list(range(-1, n_cols)))

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_facecolor("white")
        cell.set_text_props(color="black")

        if c == -1:  # row-label column (model names)
            cell.set_fontsize(LABEL_FONTSIZE)
            cell.set_text_props(ha="left", color="black", fontweight="bold")
            cell.set_facecolor("#e8e8e8" if r > 0 else "white")
            cell.PAD = 0.04
            continue
        if r == 0:  # header row (metric names)
            cell.set_fontsize(LABEL_FONTSIZE)
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(ha="center", color="black", fontweight="bold")
            cell.PAD = 0.04
            continue

        model = model_labels[r - 1]
        val = row_data[model][c]
        if val == NOT_COMPUTED:
            cell.set_fontsize(LABEL_FONTSIZE)
            cell.set_text_props(color="#888888", style="italic")
        elif model == better_model:
            cell.set_text_props(color="black", fontweight="bold")

    ax.set_title(title, color="black", fontsize=11, fontweight="bold", pad=14)
    fig.text(0.5, 0.01, footnote, ha="center", va="bottom", color="#555555", fontsize=7.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    build_table(MODEL_LABELS, RESULTS_COLS, RESULTS_ROWS, RESULTS_BETTER,
                RESULTS_TITLE, RESULTS_FOOTNOTE, OUT_DIR / "v1_vs_v2_results_table.png")
    build_table(MODEL_LABELS, ENTROPY_COLS, ENTROPY_ROWS, None,
                ENTROPY_TITLE, ENTROPY_FOOTNOTE, OUT_DIR / "v1_vs_v2_entropy_table.png")
    build_table([V2_MODEL_LABEL], ENTROPY_ONLY_COLS, ENTROPY_ONLY_ROWS, None,
                ENTROPY_ONLY_TITLE, ENTROPY_ONLY_FOOTNOTE, OUT_DIR / "v2_entropy_only_table.png")