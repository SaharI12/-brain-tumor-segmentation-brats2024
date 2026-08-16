"""
make_per_pass_table_paper.py — paper-ready (white background, black text)
version of mc_per_pass_table.png, rendered from the existing CSV so it stays
in sync with the dashboard's dark-theme table without re-running inference.

Usage:
  python make_per_pass_table_paper.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

HERE = Path(__file__).parent
CSV_PATH = HERE / "mc_per_pass_table.csv"
OUT_PATH = HERE / "mc_per_pass_table_paper.png"

with open(CSV_PATH, newline="") as f:
    reader = csv.reader(f)
    col_headers = next(reader)
    rows = list(reader)

n_passes = len(rows) - 1  # last row is Mean ± Std

HEADER_BG = "#0f3460"
SUMMARY_BG = "#e8eef7"
ROW_BG_EVEN = "#f7f9fc"
ROW_BG_ODD = "#ffffff"
GRID_COLOR = "#cccccc"
SUMMARY_TEXT = "#0f3460"

# Size the figure tightly to the actual table content (row height ~0.3in at
# this font/scale), instead of the dashboard's oversized layout — avoids
# leaving the table centered in a tall empty axes.
row_h = 0.30
fig_h = row_h * (n_passes + 2)
fig, ax = plt.subplots(figsize=(14, fig_h))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.axis("off")

tbl = ax.table(cellText=rows, colLabels=col_headers, cellLoc="center",
                bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)

for j in range(len(col_headers)):
    c = tbl[0, j]
    c.set_facecolor(HEADER_BG)
    c.set_text_props(color="white", fontweight="bold")
    c.set_edgecolor(GRID_COLOR)

for i in range(1, len(rows) + 1):
    is_summary = (i == len(rows))
    for j in range(len(col_headers)):
        c = tbl[i, j]
        c.set_edgecolor(GRID_COLOR)
        if is_summary:
            c.set_facecolor(SUMMARY_BG)
            c.set_text_props(color=SUMMARY_TEXT, fontweight="bold")
        else:
            c.set_facecolor(ROW_BG_EVEN if i % 2 == 0 else ROW_BG_ODD)
            c.set_text_props(color="black")

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.99)
plt.savefig(str(OUT_PATH), dpi=200, bbox_inches="tight", facecolor="white")
plt.close()

# The source layout reserves extra vertical space for a dashboard grid,
# leaving dead white space between the title and the table. Auto-crop to
# content (with a small margin) so this stands alone as a paper figure.
img = Image.open(OUT_PATH).convert("RGB")
bg = Image.new("RGB", img.size, "white")
diff = ImageChops.difference(img, bg)
bbox = diff.getbbox()
if bbox:
    margin = 20
    left, upper, right, lower = bbox
    left = max(0, left - margin)
    upper = max(0, upper - margin)
    right = min(img.width, right + margin)
    lower = min(img.height, lower + margin)
    img.crop((left, upper, right, lower)).save(OUT_PATH)

print(f"Saved: {OUT_PATH}")
