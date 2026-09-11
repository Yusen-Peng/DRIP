import matplotlib.pyplot as plt

# Compression settings
compression = ["1×", "4×", "8×", "10×"]

# Efficiency measurements
tflops = [8.89, 3.18, 2.23, 2.04]
kv_cache = [321.0, 105.5, 69.5, 62.5]
memory = [14.95, 14.15, 14.02, 14.00]

metrics = [
    ("Forward Compute", "TFLOPs", tflops),
    ("KV Cache", "Memory (MB)", kv_cache),
    ("GPU Memory", "Memory (GB)", memory),
]

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

for ax, (title, ylabel, values) in zip(axes, metrics):
    ax.plot(
        compression,
        values,
        marker="o",
        linewidth=2,
        markersize=6,
    )

    # Annotate exact values
    for i, value in enumerate(values):
        if ylabel == "Memory (MB)":
            label = f"{value:.1f}"
        else:
            label = f"{value:.2f}"

        ax.annotate(
            label,
            (i, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Compression Ratio", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.4,
    )

    ax.tick_params(axis="both", labelsize=9)

# Give annotation text a little breathing room
axes[0].set_ylim(0, 10)
axes[1].set_ylim(0, 360)
axes[2].set_ylim(13.8, 15.15)

plt.tight_layout()

plt.savefig("results/efficiency_analysis.pdf", bbox_inches="tight")

plt.show()
