import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection

# ============================================================
# ICLR / NeurIPS-ready style
# ============================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

    "font.size": 17,
    "axes.labelsize": 20,
    "xtick.labelsize": 12,
    "ytick.labelsize": 16,

    "axes.linewidth": 1.3,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Data
# ============================================================
methods = ["PruMerge", "PruneSID", "Fixed\nPooling", "VLVLM"]

acc_4x = [30.63, 26.25, 33.75, 33.75]
acc_8x = [29.50, 30.25, 30.75, 35.75]


# ============================================================
# Positions
# ============================================================
x_4x = np.arange(4)
x_8x = np.arange(4) + 5.2

fig, ax = plt.subplots(figsize=(10.5, 5.2))

# Muted paper-friendly colors
color_4x = "#5B7FA3"
color_8x = "#C97A40"


# ============================================================
# Bars -- NO BORDERS
# ============================================================
bars_4x = ax.bar(
    x_4x,
    acc_4x,
    width=0.70,
    color=color_4x,
    edgecolor="none",
    linewidth=0,
    zorder=2,
)

bars_8x = ax.bar(
    x_8x,
    acc_8x,
    width=0.70,
    color=color_8x,
    edgecolor="none",
    linewidth=0,
    zorder=2,
)


# ============================================================
# Custom sparse white stripes for VLVLM
#
# Unlike matplotlib hatch="/", this gives us exact control
# over BOTH spacing and thickness.
# ============================================================
def add_sparse_stripes(
    ax,
    bar,
    spacing=0.36,      # larger = sparser
    linewidth=2.5,     # thicker white stripes
    slope=4.0,
):
    x0 = bar.get_x()
    x1 = x0 + bar.get_width()

    # Important because the y-axis begins at 24
    y0 = ax.get_ylim()[0]
    y1 = bar.get_height()

    # Lines are y = slope * (x - x0) + intercept
    #
    # Sweep intercepts far enough that lines cover the entire
    # rectangle, then clip them to the bar itself.
    intercept_min = y0 - slope * (x1 - x0)
    intercept_max = y1

    intercepts = np.arange(
        intercept_min,
        intercept_max + spacing,
        spacing,
    )

    segments = []

    for b in intercepts:
        # Intersections with left/right edges
        yl = b
        yr = slope * (x1 - x0) + b

        points = []

        # Left edge
        if y0 <= yl <= y1:
            points.append((x0, yl))

        # Right edge
        if y0 <= yr <= y1:
            points.append((x1, yr))

        # Bottom edge
        xb = x0 + (y0 - b) / slope
        if x0 <= xb <= x1:
            points.append((xb, y0))

        # Top edge
        xt = x0 + (y1 - b) / slope
        if x0 <= xt <= x1:
            points.append((xt, y1))

        if len(points) >= 2:
            segments.append([points[0], points[1]])

    stripes = LineCollection(
        segments,
        colors="white",
        linewidths=linewidth,
        zorder=3,
    )

    # Clip stripes exactly to the bar
    stripes.set_clip_path(bar)
    ax.add_collection(stripes)


# Apply only to our method
add_sparse_stripes(
    ax,
    bars_4x[-1],
    spacing=0.75,
    linewidth=2.8,
)

add_sparse_stripes(
    ax,
    bars_8x[-1],
    spacing=0.75,
    linewidth=2.8,
)


# ============================================================
# X axis
# ============================================================
all_x = np.concatenate([x_4x, x_8x])

ax.set_xticks(all_x)
ax.set_xticklabels(methods + methods)

# Bold VLVLM
ax.get_xticklabels()[3].set_fontweight("bold")
ax.get_xticklabels()[7].set_fontweight("bold")


# ============================================================
# Y axis
# ============================================================
ax.set_ylabel("Accuracy", fontweight="bold")

ax.set_ylim(24, 38)
ax.set_yticks([24, 28, 32, 36])

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.25,
)

ax.set_axisbelow(True)


# ============================================================
# Group labels
# ============================================================
center_4x = np.mean(x_4x)
center_8x = np.mean(x_8x)

ax.text(
    center_4x,
    37.35,
    r"$4\times$ Compression",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
)

ax.text(
    center_8x,
    37.35,
    r"$8\times$ Compression",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
)


# ============================================================
# Divider
# ============================================================
divider_x = (x_4x[-1] + x_8x[0]) / 2

ax.axvline(
    divider_x,
    color="gray",
    linestyle="--",
    linewidth=1.2,
    alpha=0.55,
)


# ============================================================
# Value labels
# ============================================================
for i, (bar, value) in enumerate(zip(bars_4x, acc_4x)):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.20,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold" if i == 3 else "normal",
    )

for i, (bar, value) in enumerate(zip(bars_8x, acc_8x)):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.20,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold" if i == 3 else "normal",
    )


# ============================================================
# Clean appearance
# ============================================================
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(axis="x", pad=7)

plt.tight_layout()


# ============================================================
# Save
# ============================================================
plt.savefig(
    "results/synthetic_acc_plot.pdf",
    bbox_inches="tight",
)

plt.savefig(
    "results/synthetic_acc_plot.png",
    dpi=600,
    bbox_inches="tight",
)

plt.show()