import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====================================
# Config
# ====================================

csv_path = "results/motivating_example.csv"

model_col = "Model"

benchmarks = [
    "MME",
    "MM-Bench",
    "GQA",
    "MMMU",
    "TextVQA",
    "OCRBench",
    "DocVQA",
    "LLaVA-Wild",
    "MM-Vet",
]

BASELINE = "LLaVA-1.5-7B"
NO_TRAIN = "fixed pooling(no train)"
TRAINED = "fixed pooling(trained)"
PRUMERGE = "PruMerge"

# ====================================
# Load CSV
# ====================================

df = pd.read_csv(csv_path)

df.columns = df.columns.str.strip()
df[model_col] = df[model_col].str.strip()

df = df.set_index(model_col)

# ====================================
# Relative performance
# ====================================

baseline = df.loc[BASELINE, benchmarks]

trained = df.loc[TRAINED, benchmarks] / baseline * 100
no_train = df.loc[NO_TRAIN, benchmarks] / baseline * 100
prumerge = df.loc[PRUMERGE, benchmarks] / baseline * 100

# ====================================
# Plot
# ====================================

fig, ax = plt.subplots(figsize=(11, 4.5))

x = np.arange(len(benchmarks))
width = 0.55

# Outer bar: trained
ax.bar(
    x,
    trained,
    width=width,
    color="#7BC87C",
    edgecolor="black",
    linewidth=1.2,
    label="Fixed pooling (trained)",
    zorder=2,
)

# Inner bar: no train
ax.bar(
    x,
    no_train,
    width=width * 0.62,
    color="#F3D5D5",
    edgecolor="black",
    linewidth=1.0,
    hatch="//",
    label="Fixed pooling (no train)",
    zorder=3,
)

# PruMerge marker
ax.scatter(
    x,
    prumerge,
    marker="D",
    s=55,
    color="#3A86FF",
    edgecolor="black",
    linewidth=0.8,
    label="PruMerge",
    zorder=4,
)

# Upper-bound reference line
ax.axhline(
    100,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Upper bound",
    zorder=1,
)

# ====================================
# Styling
# ====================================

ax.set_ylabel("Relative performance (%)")
ax.set_xticks(x)
ax.set_xticklabels(benchmarks)

ax.set_ylim(50, 105)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    ncol=4,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.22),
)

plt.tight_layout()

plt.savefig(
    "results/motivating_example.pdf",
    bbox_inches="tight",
)

plt.savefig(
    "results/motivating_example.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
