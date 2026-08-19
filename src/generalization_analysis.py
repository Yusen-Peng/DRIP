import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# ============================================================
# Configuration
# ============================================================

INPUT_CSV = "results/generalization.csv"
OUTPUT_PDF = "results/generalization_matrix.pdf"
OUTPUT_PNG = "results/generalization_matrix.png"

BUDGETS = [4, 8, 10]


METRIC_COLS = [
    "VQAv2",
    "SQA",
    "MME",
    "MM-Bench",
    "GQA",
    "MMMU",
    "TextVQA",
    "OCRBench",
    "OCRBenchv2",
    "DocVQA",
    "ChartQAPro",
    "POPE",
    "LLaVA-Wild",
    "MM-Vet",
]


# ============================================================
# Plot style
# ============================================================

def setup_plot_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ============================================================
# Load + compute relative performance
# ============================================================

def load_data():
    df = pd.read_csv(INPUT_CSV)

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline = df.loc[
        df["Model"] == "LLaVA-1.5-7B"
    ].iloc[0]

    # Make sure metrics are numeric
    df[METRIC_COLS] = df[METRIC_COLS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    baseline_metrics = baseline[METRIC_COLS].astype(float)

    # --------------------------------------------------------
    # Relative performance to LLaVA
    # --------------------------------------------------------

    relative = df[METRIC_COLS].div(
        baseline_metrics,
        axis=1,
    )

    df["OverallScore"] = relative.mean(axis=1)

    # --------------------------------------------------------
    # Keep DRIP generalization experiments
    #
    # DRIP-4x-8x:
    #     first number  = training budget
    #     second number = inference budget
    # --------------------------------------------------------

    drip = df[
        df["Model"].str.match(r"^DRIP-\d+x-\d+x$")
    ].copy()

    extracted = drip["Model"].str.extract(
        r"DRIP-(\d+)x-(\d+)x"
    )

    drip["TrainBudget"] = extracted[0].astype(int)
    drip["TestBudget"] = extracted[1].astype(int)

    return drip


# ============================================================
# Construct matrices
# ============================================================

def build_matrices(drip):

    # --------------------------------------------------------
    # Absolute relative-performance matrix
    # --------------------------------------------------------

    performance = drip.pivot(
        index="TrainBudget",
        columns="TestBudget",
        values="OverallScore",
    )

    performance = performance.loc[
        BUDGETS,
        BUDGETS,
    ]

    # --------------------------------------------------------
    # Generalization gap
    #
    # For each inference budget:
    #
    #     score(train=X, test=Y)
    #       -
    #     score(train=Y, test=Y)
    #
    # Therefore diagonal = 0.
    # --------------------------------------------------------

    gap = performance.copy()

    for test_budget in BUDGETS:

        matched_score = performance.loc[
            test_budget,
            test_budget,
        ]

        gap[test_budget] = (
            performance[test_budget]
            - matched_score
        )

    # Convert to percentage points
    gap *= 100.0

    return performance, gap


# ============================================================
# Plot
# ============================================================

def plot_matrix(performance, gap):

    setup_plot_style()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.5),
    )

    # ========================================================
    # Panel 1: Relative performance
    # ========================================================

    ax = axes[0]

    perf_percent = performance.to_numpy() * 100

    im1 = ax.imshow(
        perf_percent,
        cmap="Reds",
        aspect="equal",
    )

    for i in range(len(BUDGETS)):
        for j in range(len(BUDGETS)):

            value = perf_percent[i, j]

            weight = (
                "bold"
                if i == j
                else "normal"
            )

            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=13,
                fontweight=weight,
            )

    ax.set_xticks(range(len(BUDGETS)))
    ax.set_yticks(range(len(BUDGETS)))

    ax.set_xticklabels(
        [f"{x}×" for x in BUDGETS]
    )

    ax.set_yticklabels(
        [f"{x}×" for x in BUDGETS]
    )

    ax.set_xlabel("Inference Compression Ratio")
    ax.set_ylabel("Training Compression Ratio")

    ax.set_title(
        "Overall Performance",
        pad=12,
        fontweight="bold",
    )

    cbar1 = fig.colorbar(
        im1,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar1.set_label(
        "Relative Performance (%)"
    )

    # ========================================================
    # Panel 2: Generalization gap
    # ========================================================

    ax = axes[1]

    gap_values = gap.to_numpy()

    # Symmetric color range around zero
    max_abs = np.nanmax(np.abs(gap_values))

    im2 = ax.imshow(
        gap_values,
        cmap="RdBu",
        vmin=-max_abs,
        vmax=max_abs,
        aspect="equal",
    )

    # One text color per inference-budget column
    column_text_colors = [
        "#1f77b4",  # 4×
        "#2ca02c",  # 8×
        "#9467bd",  # 10×
    ]

    for i in range(len(BUDGETS)):
        for j in range(len(BUDGETS)):

            value = gap_values[i, j]

            weight = (
                "bold"
                if i == j
                else "normal"
            )

            ax.text(
                j,
                i,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=13,
                fontweight=weight,
                color=column_text_colors[j],
                zorder=5,
            )

    for j in range(len(BUDGETS)):

        reference_row = j

        # Slight horizontal offset from text
        x_arrow = j + 0.30

        # Upward comparison
        if reference_row > 0:
            ax.annotate(
                "",
                xy=(x_arrow, -0.35),
                xytext=(x_arrow, reference_row - 0.35),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=column_text_colors[j],
                    linewidth=1.4,
                    linestyle="--",
                    alpha=0.75,
                    mutation_scale=10,
                ),
                zorder=4,
            )

        # Downward comparison
        if reference_row < len(BUDGETS) - 1:
            ax.annotate(
                "",
                xy=(x_arrow, len(BUDGETS) - 1 + 0.35),
                xytext=(x_arrow, reference_row + 0.35),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=column_text_colors[j],
                    linewidth=1.4,
                    linestyle="--",
                    alpha=0.75,
                    mutation_scale=10,
                ),
                zorder=4,
            )

    ax.set_xticks(range(len(BUDGETS)))
    ax.set_yticks(range(len(BUDGETS)))

    ax.set_xticklabels(
        [f"{x}×" for x in BUDGETS]
    )

    ax.set_yticklabels(
        [f"{x}×" for x in BUDGETS]
    )

    ax.set_xlabel("Inference Compression Ratio")
    ax.set_ylabel("Training Compression Ratio")

    ax.set_title(
        "Gap from Matched-Budget Training",
        pad=12,
        fontweight="bold",
    )

    cbar2 = fig.colorbar(
        im2,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar2.set_label(
        "Performance Difference (pp)"
    )

    # ========================================================
    # Outline matched-budget diagonal
    # ========================================================

    for ax in axes:
        for i in range(len(BUDGETS)):

            rect = plt.Rectangle(
                (i - 0.5, i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=1.8,
            )

            ax.add_patch(rect)

        # Remove external spines
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()

    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=400,
        bbox_inches="tight",
    )

    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    drip = load_data()

    performance, gap = build_matrices(drip)

    print("\nOverall relative performance:")
    print(performance)

    print("\nGeneralization gap (percentage points):")
    print(gap)

    plot_matrix(
        performance,
        gap,
    )
