import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

VICUNA_CSV = "results/full_7B_last.csv"
QWEN_CSV   = "results/qwen14B_full_last.csv"

OCR_BENCHMARKS = [
    "TextVQA",
    "OCRBench",
    "OCRBenchv2",
    "DocVQA",
    "ChartQAPro",
]

COMPRESSION_LEVELS = ["4x", "8x", "10x"]


# ============================================================
# Load results
# ============================================================

vicuna = pd.read_csv(VICUNA_CSV)
qwen = pd.read_csv(QWEN_CSV)


def find_row(df, method, compression=None):
    """
    Find a model row robustly from the Model column.
    """
    names = df["Model"].str.lower()

    if method == "baseline":
        # Assumes the first row is the uncompressed model.
        return df.iloc[0]

    mask = names.str.contains(method.lower(), regex=False)

    if compression is not None:
        mask &= names.str.contains(compression.lower(), regex=False)

    rows = df[mask]

    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one row for "
            f"method={method}, compression={compression}, "
            f"but found {len(rows)}:\n{rows['Model'].tolist()}"
        )

    return rows.iloc[0]


def compute_drip_gain(df):
    """
    Compute VLVLM - Fixed Pooling for each compression level.

    Each benchmark is first normalized relative to the
    corresponding uncompressed model:

        normalized_score = compressed_score / baseline_score * 100

    The five normalized OCR benchmark scores are then averaged.

    Returns:
        fixed_scores
        drip_scores
        gains
    """
    baseline = find_row(df, "baseline")

    fixed_scores = []
    drip_scores = []
    gains = []

    for level in COMPRESSION_LEVELS:

        fixed = find_row(df, "fixed pooling", level)
        drip = find_row(df, "VLVLM", level)

        baseline_values = baseline[OCR_BENCHMARKS].astype(float).values
        fixed_values = fixed[OCR_BENCHMARKS].astype(float).values
        drip_values = drip[OCR_BENCHMARKS].astype(float).values

        # Normalize relative to the uncompressed model
        fixed_normalized = 100 * fixed_values / baseline_values
        drip_normalized = 100 * drip_values / baseline_values

        fixed_avg = fixed_normalized.mean()
        drip_avg = drip_normalized.mean()

        fixed_scores.append(fixed_avg)
        drip_scores.append(drip_avg)
        gains.append(drip_avg - fixed_avg)

    return (
        np.array(fixed_scores),
        np.array(drip_scores),
        np.array(gains),
    )


vicuna_fixed, vicuna_drip, vicuna_gain = compute_drip_gain(vicuna)
qwen_fixed, qwen_drip, qwen_gain = compute_drip_gain(qwen)


# ============================================================
# Print values for sanity checking
# ============================================================

print("\nVicuna-7B")
for level, f, d, g in zip(
    COMPRESSION_LEVELS, vicuna_fixed, vicuna_drip, vicuna_gain
):
    print(
        f"{level:>3}: Fixed={f:.2f}, "
        f"VLVLM={d:.2f}, Δ={g:+.2f}"
    )

print("\nQwen2.5-14B")
for level, f, d, g in zip(
    COMPRESSION_LEVELS, qwen_fixed, qwen_drip, qwen_gain
):
    print(
        f"{level:>3}: Fixed={f:.2f}, "
        f"VLVLM={d:.2f}, Δ={g:+.2f}"
    )


# ============================================================
# Plot
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,

    # Clean conference-paper look
    "axes.spines.top": False,
    "axes.spines.right": False,

    # PDF-compatible fonts
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.35, 2.25))

x = np.arange(len(COMPRESSION_LEVELS))

ax.plot(
    x,
    vicuna_gain,
    marker="o",
    markersize=5,
    linewidth=1.6,
    label="Vicuna-7B",
)

ax.plot(
    x,
    qwen_gain,
    marker="s",
    markersize=5,
    linewidth=1.6,
    label="Qwen2.5-14B",
)

# Zero = VLVLM and Fixed are equivalent
ax.axhline(
    0,
    linewidth=0.8,
    linestyle="--",
    alpha=0.6,
)

ax.set_xticks(x)
ax.set_xticklabels(["4×", "8×", "10×"])

ax.set_xlabel("Image Token Compression")
ax.set_ylabel("VLVLM − Fixed (normalized pts.)")

ax.legend(
    frameon=False,
    loc="best",
)

ax.grid(
    axis="y",
    linewidth=0.5,
    alpha=0.25,
)

ax.margins(x=0.10)

fig.tight_layout(pad=0.4)

plt.savefig(
    "results/llm_scaling_ocr.pdf",
    bbox_inches="tight",
    dpi=300,
)

plt.show()

vicuna = pd.read_csv(VICUNA_CSV)
qwen   = pd.read_csv(QWEN_CSV)


def find_row(df, method, compression=None):

    names = df["Model"].str.lower()

    if method == "baseline":
        return df.iloc[0]

    mask = names.str.contains(method.lower(), regex=False)

    if compression is not None:
        mask &= names.str.contains(compression.lower(), regex=False)

    rows = df[mask]

    if len(rows) != 1:
        raise ValueError(
            f"Expected one row for {method}, {compression}, "
            f"found: {rows['Model'].tolist()}"
        )

    return rows.iloc[0]


# ------------------------------------------------------------
# Common normalization reference:
# uncompressed Vicuna-7B
# ------------------------------------------------------------

vicuna_baseline = find_row(vicuna, "baseline")
reference = vicuna_baseline[OCR_BENCHMARKS].astype(float).values


def scaling_gain(method):

    gains = []

    for level in COMPRESSION_LEVELS:

        vicuna_row = find_row(vicuna, method, level)
        qwen_row   = find_row(qwen, method, level)

        vicuna_scores = (
            vicuna_row[OCR_BENCHMARKS]
            .astype(float)
            .values
        )

        qwen_scores = (
            qwen_row[OCR_BENCHMARKS]
            .astype(float)
            .values
        )

        # Improvement caused by scaling the LLM
        benchmark_gains = (
            100 * (qwen_scores - vicuna_scores) / reference
        )

        gains.append(benchmark_gains.mean())

    return np.array(gains)


fixed_gain = scaling_gain("fixed pooling")
drip_gain  = scaling_gain("VLVLM")


# ------------------------------------------------------------
# Print numbers
# ------------------------------------------------------------

print("\nScaling gain: Vicuna-7B -> Qwen2.5-14B")

for level, fixed, drip in zip(
    COMPRESSION_LEVELS,
    fixed_gain,
    drip_gain,
):
    print(
        f"{level}: "
        f"Fixed={fixed:+.2f}, "
        f"VLVLM={drip:+.2f}, "
        f"VLVLM advantage={drip-fixed:+.2f}"
    )


# ------------------------------------------------------------
# ICLR / NeurIPS-style plot
# ------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,

    "axes.spines.top": False,
    "axes.spines.right": False,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


fig, ax = plt.subplots(figsize=(3.35, 2.15))

x = np.arange(len(COMPRESSION_LEVELS))
width = 0.34


ax.bar(
    x - width / 2,
    fixed_gain,
    width,
    label="Fixed Pooling",
)

ax.bar(
    x + width / 2,
    drip_gain,
    width,
    label="VLVLM",
)


# Zero reference
ax.axhline(
    0,
    linewidth=0.8,
    linestyle="--",
    alpha=0.6,
)


ax.set_xticks(x)
ax.set_xticklabels(["4×", "8×", "10×"])

ax.set_xlabel("Image Token Compression")
ax.set_ylabel("Gain from LLM Scaling\n(normalized pts.)")

ax.legend(
    frameon=False,
    ncol=2,
    loc="upper center",
)

ax.grid(
    axis="y",
    linewidth=0.5,
    alpha=0.25,
)

ax.set_axisbelow(True)

fig.tight_layout(pad=0.4)


plt.savefig(
    "results/llm_scaling_gain_ocr.pdf",
    bbox_inches="tight",
    dpi=300,
)
plt.show()