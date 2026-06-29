"""
Generate a clean 3D U-Net architecture diagram (paper style, no legend box).
Saves to exploration_output/architecture.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from pathlib import Path

OUT = Path("exploration_output/architecture.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
BG     = "#ffffff"
ENC_FC = "#d4e6f1"
ENC_EC = "#2471a3"
BOT_FC = "#fadbd8"
BOT_EC = "#c0392b"
DEC_FC = "#d5f5e3"
DEC_EC = "#1e8449"
IO_FC  = "#f2f3f4"
IO_EC  = "#555555"
SKIP_C = "#e67e22"
FLOW_C = "#444444"
DROP_C = "#8e44ad"

F      = 32
BOT_CH = min(F * 16, 320)

# ── Block geometry ────────────────────────────────────────────────────────────
# U-Net U-shape: encoder descends left→right, decoder ascends
# 5 depth levels × 9 columns (0=enc0 … 4=bottleneck, 5=dec3 … 8=dec0)
#
# Column x-centres
COL_W  = 3.0
ROW_H  = 2.4
BW     = 2.4   # block width
BH     = 0.80  # block height

def cx(col): return col * COL_W          # column x centre
def cy(row): return -row * ROW_H         # row y centre (row 0 at top)

# (col, row, title, sub, fc, ec)
BLOCKS = [
    # Encoder
    (0, 0, f"{F} ch", "ConvBlock × 2",       ENC_FC, ENC_EC),
    (1, 1, f"{F*2} ch", "ConvBlock × 2",     ENC_FC, ENC_EC),
    (2, 2, f"{F*4} ch", "ConvBlock × 2",     ENC_FC, ENC_EC),
    (3, 3, f"{F*8} ch", "ConvBlock × 2",     ENC_FC, ENC_EC),
    # Bottleneck
    (4, 4, f"{BOT_CH} ch", "Conv × 2  +  Dropout3d", BOT_FC, BOT_EC),
    # Decoder
    (5, 3, f"{F*8} ch", "ConvBlock × 2  +  Dropout3d", DEC_FC, DEC_EC),
    (6, 2, f"{F*4} ch", "ConvBlock × 2  +  Dropout3d", DEC_FC, DEC_EC),
    (7, 1, f"{F*2} ch", "ConvBlock × 2",     DEC_FC, DEC_EC),
    (8, 0, f"{F} ch",   "ConvBlock × 2",     DEC_FC, DEC_EC),
]

# Spatial resolution labels per row
SPATIAL = {
    0: "160×208×160",
    1: "80×104×80",
    2: "40×52×40",
    3: "20×26×20",
    4: "10×13×10",
}

# Input / output
IO_BLOCKS = [
    (-1, 0, "4 ch",   "Input\n(T1n · T1c · T2w · T2f)", IO_FC, IO_EC),
    ( 9, 0, "4 ch",   "Output\n(class logits)",          IO_FC, IO_EC),
]

# ── Figure ────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 26, 15
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

XMIN = cx(-1) - BW / 2 - 0.6
XMAX = cx(9)  + BW / 2 + 0.6
YMIN = cy(4)  - BH / 2 - 1.8
YMAX = cy(0)  + BH / 2 + 1.8
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)
ax.set_aspect("equal")
ax.axis("off")


# ── Draw a block ─────────────────────────────────────────────────────────────
def draw_block(col, row, title, sub, fc, ec, lw=1.6):
    x, y = cx(col), cy(row)
    rect = mpatches.FancyBboxPatch(
        (x - BW / 2, y - BH / 2), BW, BH,
        boxstyle="round,pad=0.05",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x, y + 0.14, title, ha="center", va="center",
            fontsize=9, fontweight="bold", color="#111", zorder=4)
    ax.text(x, y - 0.16, sub, ha="center", va="center",
            fontsize=7, color="#333", zorder=4)


for args in BLOCKS + IO_BLOCKS:
    draw_block(*args)

# ── Spatial resolution annotations (right side of encoder, left of decoder) ──
for row, label in SPATIAL.items():
    # Find the encoder column at this row
    enc_cols = [col for col, r, *_ in BLOCKS if r == row and col <= 4]
    dec_cols = [col for col, r, *_ in BLOCKS if r == row and col > 4]
    if enc_cols:
        x = cx(min(enc_cols)) - BW / 2 - 0.1
        ax.text(x, cy(row), label, ha="right", va="center",
                fontsize=7.5, color="#666", fontstyle="italic")


# ── Arrows: encoder flow (diagonal down-right) ───────────────────────────────
def angled_arrow(x0, y0, x1, y1, color, lw=1.4, label=None, label_side="left"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>", color=color, lw=lw,
                    mutation_scale=12,
                    connectionstyle="angle,angleA=-90,angleB=180,rad=5",
                ),
                zorder=2)
    if label:
        xm = (x0 + x1) / 2 + (-0.55 if label_side == "left" else 0.55)
        ym = (y0 + y1) / 2
        ax.text(xm, ym, label, ha="center", va="center",
                fontsize=6.5, color="#777", style="italic",
                bbox=dict(fc=BG, ec="none", pad=1))


def angled_arrow_dec(x0, y0, x1, y1, color, lw=1.4, label=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>", color=color, lw=lw,
                    mutation_scale=12,
                    connectionstyle="angle,angleA=0,angleB=-90,rad=5",
                ),
                zorder=2)
    if label:
        xm = (x0 + x1) / 2 + 0.55
        ym = (y0 + y1) / 2
        ax.text(xm, ym, label, ha="center", va="center",
                fontsize=6.5, color="#777", style="italic",
                bbox=dict(fc=BG, ec="none", pad=1))


# Input → enc0
ax.annotate("", xy=(cx(0) - BW / 2, cy(0)), xytext=(cx(-1) + BW / 2, cy(-1 + 1)),
            arrowprops=dict(arrowstyle="-|>", color=FLOW_C, lw=1.4, mutation_scale=12),
            zorder=2)

# Encoder down-steps
enc_rows = [0, 1, 2, 3]
for i in range(len(enc_rows) - 1):
    col, row     = i, enc_rows[i]
    col1, row1   = i + 1, enc_rows[i + 1]
    angled_arrow(cx(col), cy(row) - BH / 2,
                 cx(col1) - BW / 2, cy(row1),
                 ENC_EC, lw=1.4, label="stride-2\nDownConv")

# enc3 → bottleneck
angled_arrow(cx(3), cy(3) - BH / 2,
             cx(4) - BW / 2, cy(4),
             BOT_EC, lw=1.6, label="stride-2\nDownConv")

# Bottleneck → dec3
angled_arrow_dec(cx(4) + BW / 2, cy(4),
                 cx(5) - BW / 2, cy(3),
                 BOT_EC, lw=1.6, label="upsample\n+ concat")

# Decoder up-steps
dec_start_cols = [5, 6, 7]
dec_rows       = [3, 2, 1]
for i in range(len(dec_start_cols) - 1):
    col, row   = dec_start_cols[i], dec_rows[i]
    col1, row1 = dec_start_cols[i + 1], dec_rows[i + 1]
    angled_arrow_dec(cx(col) + BW / 2, cy(row),
                     cx(col1) - BW / 2, cy(row1),
                     DEC_EC, lw=1.4, label="upsample\n+ concat")

# dec1 → dec0
angled_arrow_dec(cx(7) + BW / 2, cy(1),
                 cx(8) - BW / 2, cy(0),
                 DEC_EC, lw=1.4, label="upsample\n+ concat")

# dec0 → output  (1×1 conv label)
ax.annotate("", xy=(cx(9) - BW / 2, cy(0)), xytext=(cx(8) + BW / 2, cy(0)),
            arrowprops=dict(arrowstyle="-|>", color=FLOW_C, lw=1.4, mutation_scale=12),
            zorder=2)
xm = (cx(8) + BW / 2 + cx(9) - BW / 2) / 2
ax.text(xm, cy(0) + 0.28, "Conv3d 1×1×1", ha="center", va="bottom",
        fontsize=6.5, color="#777", style="italic")

# ── Skip connections ──────────────────────────────────────────────────────────
skip_pairs = [(0, 8), (1, 7), (2, 6), (3, 5)]   # (enc_col, dec_col)
for enc_col, dec_col in skip_pairs:
    row = enc_col   # enc_col == row for encoder
    y_skip = cy(row) + BH / 2 + 0.22
    x0 = cx(enc_col) + BW / 2
    x1 = cx(dec_col) - BW / 2
    ax.annotate("", xy=(x1, y_skip), xytext=(x0, y_skip),
                arrowprops=dict(
                    arrowstyle="-|>", color=SKIP_C, lw=1.8,
                    mutation_scale=11, linestyle="dashed",
                ),
                zorder=2)
    xm = (x0 + x1) / 2
    ax.text(xm, y_skip + 0.13, "skip connection (concat)",
            ha="center", va="bottom", fontsize=6.5, color=SKIP_C)


# ── MC Dropout highlights ─────────────────────────────────────────────────────
dropout_blocks = [(4, 4), (5, 3), (6, 2)]   # (col, row) — bottleneck + dec3 + dec2
for col, row in dropout_blocks:
    x, y = cx(col), cy(row)
    rect = mpatches.FancyBboxPatch(
        (x - BW / 2 - 0.07, y - BH / 2 - 0.07), BW + 0.14, BH + 0.14,
        boxstyle="round,pad=0.04",
        linewidth=2.0, edgecolor=DROP_C, facecolor="none",
        linestyle=(0, (5, 3)), zorder=5,
    )
    ax.add_patch(rect)

# MC Dropout label — bracket on the right side
x_drop = cx(6) + BW / 2 + 0.2
y_top  = cy(3) + BH / 2 + 0.07
y_bot  = cy(4) - BH / 2 - 0.07
# vertical line + ticks
ax.plot([x_drop + 0.1, x_drop + 0.1], [y_bot, y_top], color=DROP_C, lw=1.5, zorder=4)
ax.plot([x_drop + 0.0, x_drop + 0.1], [y_top, y_top], color=DROP_C, lw=1.5, zorder=4)
ax.plot([x_drop + 0.0, x_drop + 0.1], [y_bot, y_bot], color=DROP_C, lw=1.5, zorder=4)
ax.text(x_drop + 0.25, (y_top + y_bot) / 2,
        "MC Dropout\n(active at\ninference)",
        ha="left", va="center", fontsize=7.5, color=DROP_C, fontweight="bold")

# ── Section labels ────────────────────────────────────────────────────────────
ax.text(cx(1.5), cy(0) + BH / 2 + 1.1, "Encoder",
        ha="center", va="center", fontsize=12, fontweight="bold", color=ENC_EC)
ax.text(cx(4),   cy(4) - BH / 2 - 1.1, "Bottleneck",
        ha="center", va="center", fontsize=12, fontweight="bold", color=BOT_EC)
ax.text(cx(6.5), cy(0) + BH / 2 + 1.1, "Decoder",
        ha="center", va="center", fontsize=12, fontweight="bold", color=DEC_EC)

# ── ConvBlock annotation ─────────────────────────────────────────────────────
ax.text((XMIN + cx(-1) - BW / 2) / 2 + 0.2, cy(4),
        "ConvBlock:\nConv3d 3×3×3\n→ InstanceNorm3d\n→ LeakyReLU(0.01)\n(repeated × 2)",
        ha="center", va="center", fontsize=7, color="#444",
        bbox=dict(fc="#f8f9fa", ec="#ccc", boxstyle="round,pad=0.4", lw=0.8))

# ── Title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    "3D U-Net with Monte Carlo Dropout  —  BraTS 2024 GLI\n"
    f"init_features = {F}   |   21.7 M parameters   |   "
    "Input: (B, 4, 160, 208, 160)  →  Output: (B, 4, 160, 208, 160)",
    fontsize=11, fontweight="bold", color="#111", pad=10,
)

plt.tight_layout()
plt.savefig(str(OUT), dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.close()
print(f"Saved: {OUT}")