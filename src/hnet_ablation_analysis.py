import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ---------------------------------------------------------
# Data
# ---------------------------------------------------------
benchmarks = ["TextVQA", "OCRBench", "OCRBenchv2", "DocVQA", "ChartQAPro"]

drip = np.array([55.84, 16.50, 20.90, 20.39, 10.91])
hnet = np.array([54.28, 14.90, 20.40, 18.28, 11.19])

delta = drip - hnet

# ---------------------------------------------------------
# Paper-style plotting setup
# ---------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---------------------------------------------------------
# Figure
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 2.25))

x = np.arange(len(benchmarks))

# Slightly distinguish positive / negative bars
bar_colors = [
    "#4C78A8" if d >= 0 else "#B85C5C"
    for d in delta
]

bars = ax.bar(
    x,
    delta,
    width=0.56,
    color=bar_colors,
    edgecolor="black",   # hatch uses edgecolor
    linewidth=0.6,
    hatch="////",        # diagonal lines
    zorder=3,
)

# ---------------------------------------------------------
# Axes
# ---------------------------------------------------------
ax.axhline(
    0,
    color="black",
    linewidth=0.75,
    zorder=4,
)

ax.set_xticks(x)
ax.set_xticklabels(benchmarks)

ax.set_ylabel(r"$\Delta$ score (MLP $-$ H-Net)")

# Subtle horizontal grid only
ax.yaxis.grid(
    True,
    linewidth=0.45,
    alpha=0.25,
    zorder=0,
)

ax.xaxis.grid(False)

# Remove unnecessary borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(0.7)
ax.spines["bottom"].set_linewidth(0.7)

# Give annotations enough room
ax.set_ylim(-0.60, 2.36)

# ---------------------------------------------------------
# Value labels
# ---------------------------------------------------------
for bar, d in zip(bars, delta):

    x_pos = bar.get_x() + bar.get_width() / 2

    if d >= 0:
        y_pos = d + 0.06
        va = "bottom"
    else:
        y_pos = d - 0.06
        va = "top"

    ax.text(
        x_pos,
        y_pos,
        f"{d:+.2f}",
        ha="center",
        va=va,
        fontsize=8,
    )

# ---------------------------------------------------------
# Spacing
# ---------------------------------------------------------
ax.margins(x=0.045)

plt.tight_layout(pad=0.4)

plt.savefig(
    "results/boundary_predictor_ocr_HNet_ablation_analysis.pdf",
    bbox_inches="tight",
    pad_inches=0.02,
)

plt.show()