import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = "results/motivating_example.csv"
out_prefix = "results/motivating_example"

MODEL_COL = "Model"
BASELINE = "LLaVA-1.5-7B"
NO_TRAIN = "fixed pooling(no train)"
TRAINED = "fixed pooling(trained)"
PRUMERGE = "PruMerge"

benchmarks = [
   "VQAv2","MME","MM-Bench","GQA","MMMU","TextVQA","OCRBench","OCRBenchv2","ChartQAPro","POPE"
]

# =========================
# Load + clean
# =========================

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

df[MODEL_COL] = (
    df[MODEL_COL]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

for c in df.columns:
    if c != MODEL_COL:
        df[c] = pd.to_numeric(df[c], errors="raise")

df = df.set_index(MODEL_COL)

required_rows = [BASELINE, NO_TRAIN, TRAINED, PRUMERGE]
missing_rows = [r for r in required_rows if r not in df.index]
if missing_rows:
    raise ValueError(f"Missing rows: {missing_rows}\nAvailable rows: {df.index.tolist()}")

missing_cols = [c for c in benchmarks + ["TFLOPs"] if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}\nAvailable columns: {df.columns.tolist()}")

# =========================
# Relative performance
# =========================

baseline = df.loc[BASELINE, benchmarks]

rel = df.loc[[NO_TRAIN, TRAINED, PRUMERGE], benchmarks].div(baseline, axis=1) * 100

no_train = rel.loc[NO_TRAIN]
trained = rel.loc[TRAINED]
prumerge = rel.loc[PRUMERGE]

# print(f"trained: {trained}")
# print(f"no_train: {no_train}")
# print(f"prumerge: {prumerge}")

gain = trained - no_train

# Sort by training gain, strongest visual first
benchmarks = gain.sort_values(ascending=False).index.tolist()

no_train = no_train[benchmarks]
trained = trained[benchmarks]
prumerge = prumerge[benchmarks]
gain = gain[benchmarks]

# Sanity check
bad = gain[gain < 0]
if len(bad) > 0:
    print("Warning: trained is worse than no-train on:")
    print(bad)

# =========================
# Plot
# =========================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

fig, ax = plt.subplots(figsize=(11.5, 4.2))

x = np.arange(len(benchmarks))

# two visual groups per benchmark:
# left  = PruMerge
# right = fixed pooling family, overlapped
pru_w = 0.28
fix_w = 0.44

pru_x = x - 0.25
fix_x = x + 0.18

colors = {
    "prumerge": "#5B8DB8",   # calm blue
    "trained": "#8FBF7F",   # soft sage green
    "no_train": "#D9A441",  # muted amber
}

# PruMerge standalone column
ax.bar(
    pru_x,
    prumerge,
    width=pru_w,
    color=colors["prumerge"],
    edgecolor="black",
    linewidth=0.8,
    label="PruMerge",
    zorder=3,
)

# Fixed pooling trained: outer/taller bar
ax.bar(
    fix_x,
    trained,
    width=fix_w,
    color=colors["trained"],
    edgecolor="black",
    linewidth=0.9,
    label="Fixed pooling + training",
    zorder=2,
)

# Fixed pooling no-train: inner/shorter bar
ax.bar(
    fix_x,
    no_train,
    width=fix_w * 0.58,
    color=colors["no_train"],
    edgecolor="black",
    linewidth=0.8,
    label="Fixed pooling, no training",
    zorder=4,
)

# Upper-bound reference
ax.axhline(
    100,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="LLaVA-1.5 upper bound",
    zorder=1,
)

# Training gain annotations on fixed pooling family
for i, g in enumerate(gain):
    ax.text(
        fix_x[i],
        trained.iloc[i] + 1.0,
        f"+{g:.1f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#E74F4F",
        fontweight="bold"
    )

ax.set_ylabel("Relative performance to LLaVA-1.5-7B (%)")
ax.set_xticks(x)
ax.set_xticklabels(benchmarks, rotation=25, ha="right")

ymin = max(45, np.floor(min(no_train.min(), trained.min(), prumerge.min()) / 5) * 5 - 5)
ax.set_ylim(ymin, 106)

ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    ncol=4,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.22),
    columnspacing=1.2,
    handletextpad=0.5,
)

avg_gain = gain.mean()
ax.text(
    0.01,
    0.94,
    f"Avg. gain from training: +{avg_gain:.1f} pts",
    transform=ax.transAxes,
    fontsize=10.5,
    fontweight="bold"
)

plt.tight_layout()
# save pngs
plt.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
# save pdfs 
plt.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
plt.show()

