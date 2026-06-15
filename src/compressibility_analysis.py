import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


csv_path = "results/full_7B_last.csv"
df = pd.read_csv(csv_path)

baseline_name = "LLaVA-1.5-7B"
model_col = "Model"

# Benchmarks only, exclude TFLOPs
benchmarks = [
    c for c in df.columns
    if c not in [model_col, "TFLOPs"]
]

baseline = df[df[model_col] == baseline_name].iloc[0]

compressed_df = df[df[model_col] != baseline_name].copy()

# -----------------------
# 2. Compute retention (%)
# -----------------------
retention = compressed_df.copy()

for b in benchmarks:
    retention[b] = 100.0 * compressed_df[b] / baseline[b]

retention = retention[[model_col] + benchmarks]

# -----------------------
# 3. Sort benchmarks by avg retention
#    high = more compressible
# -----------------------
avg_retention = retention[benchmarks].mean(axis=0)
sorted_benchmarks = avg_retention.sort_values(ascending=False).index.tolist()

heatmap_df = retention.set_index(model_col)[sorted_benchmarks].T

# Optional: nicer column names
rename_cols = {
    "fixed pooling-4x": "Fixed-4×",
    "PruMerge-4x": "PruMerge-4×",
    "DRIP-4x": "DRIP-4×",
    "fixed pooling-8x": "Fixed-8×",
    "PruMerge-8x": "PruMerge-8×",
    "DRIP-8x": "DRIP-8×",
    "fixed pooling-10x": "Fixed-10×",
    "PruMerge-10x": "PruMerge-10×",
    "DRIP-10x": "DRIP-10×",
}
heatmap_df = heatmap_df.rename(columns=rename_cols)

# -----------------------
# 4. Draw retention heatmap
# -----------------------
plt.figure(figsize=(13, 7))

ax = sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".1f",
    cmap="RdYlBu",
    center=100,
    vmin=75,
    vmax=105,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "Retention vs LLaVA-1.5-7B (%)"},
)

ax.set_title(
    "Benchmark Compressibility Heatmap",
    fontsize=16,
    fontweight="bold",
    pad=14,
)
ax.set_xlabel("Compressed method")
ax.set_ylabel("Benchmark, sorted by average retention")

plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(f"results/retention_heatmap_{csv_path.split('/')[-1].replace('.csv', '')}.pdf", bbox_inches="tight")
plt.savefig(f"results/retention_heatmap_{csv_path.split('/')[-1].replace('.csv', '')}.png", dpi=300, bbox_inches="tight")
plt.show()
