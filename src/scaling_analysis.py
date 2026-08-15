import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# ============================================================
# Configuration
# ============================================================

CSV_ID = "scaling"
INPUT_CSV = f"results/{CSV_ID}.csv"

OUTPUT_PDF = f"results/{CSV_ID}_sft_scaling_combined.pdf"
OUTPUT_PNG = f"results/{CSV_ID}_sft_scaling_combined.png"

FULL_SFT_SIZE_K = 665.0


# ============================================================
# Benchmark groups
# ============================================================

ALL_BENCHMARKS = [
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

OCR_BENCHMARKS = [
    "TextVQA",
    "OCRBench",
    "OCRBenchv2",
    "DocVQA",
    "ChartQAPro",
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
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ============================================================
# Model utilities
# ============================================================

def normalize_model_name(name):
    name = str(name).strip()

    name = name.replace("fixed-pooling", "fixed pooling")
    name = name.replace("Fixed pooling", "fixed pooling")
    name = name.replace("Fixed-pooling", "fixed pooling")

    return name


def get_category(model_name):
    name = model_name.lower()

    if name.startswith("llava"):
        return "LLaVA"
    elif name.startswith("fixed pooling"):
        return "Fixed pooling"
    elif name.startswith("drip"):
        return "DRIP"

    return "Other"


def get_compression(model_name):
    name = model_name.lower()

    if name.startswith("llava"):
        return "LLaVA"

    for ratio in ["4x", "8x", "10x"]:
        if ratio in name:
            return ratio

    return ""


def pretty_model_name(model_name):
    category = get_category(model_name)
    compression = get_compression(model_name)

    if category == "LLaVA":
        return "LLaVA"

    if category == "DRIP":
        return f"DRIP-{compression}"

    if category == "Fixed pooling":
        return f"Fixed-{compression}"

    return model_name


# ============================================================
# Data processing
# ============================================================

def prepare_data(df):
    df = df.copy()

    df["Model"] = df["Model"].apply(normalize_model_name)

    df["data_scale"] = pd.to_numeric(
        df["data_scale"],
        errors="coerce",
    )

    for col in ALL_BENCHMARKS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Baseline:
    # fully trained, uncompressed LLaVA
    # --------------------------------------------------------

    baseline_mask = (
        (df["Model"] == "LLaVA-1.5-7B")
        &
        np.isclose(df["data_scale"], 1.0)
    )

    baseline_rows = df.loc[baseline_mask]

    if len(baseline_rows) != 1:
        raise ValueError(
            "Expected exactly one fully-trained "
            "LLaVA-1.5-7B row."
        )

    baseline_row = baseline_rows.iloc[0]

    # --------------------------------------------------------
    # Normalize every benchmark by full-data LLaVA
    # --------------------------------------------------------

    relative = df[ALL_BENCHMARKS].div(
        baseline_row[ALL_BENCHMARKS],
        axis=1,
    )

    # --------------------------------------------------------
    # Aggregate scores
    # --------------------------------------------------------

    df["OverallScore"] = relative[
        ALL_BENCHMARKS
    ].mean(
        axis=1,
        skipna=True,
    )

    df["OCRScore"] = relative[
        OCR_BENCHMARKS
    ].mean(
        axis=1,
        skipna=True,
    )

    # --------------------------------------------------------
    # Actual SFT dataset size
    # --------------------------------------------------------

    df["SFT_Data_K"] = (
        df["data_scale"]
        * FULL_SFT_SIZE_K
    )

    df["Category"] = df["Model"].apply(
        get_category
    )

    df["Compression"] = df["Model"].apply(
        get_compression
    )

    return df


# ============================================================
# Single panel
# ============================================================

def plot_scaling_panel(
    ax,
    df,
    score_col,
    title,
    ylabel,
):
    colors = {
        "LLaVA": "#6E6E6E",
        "Fixed pooling": "#F28E2B",
        "DRIP": "#E15759",
    }

    markers = {
        "LLaVA": "o",
        "4x": "s",
        "8x": "^",
        "10x": "D",
    }

    linestyles = {
        "LLaVA": "-",
        "4x": "-",
        "8x": "--",
        "10x": ":",
    }

    model_order = [
        "LLaVA-1.5-7B",

        "fixed pooling-4x",
        "fixed pooling-8x",
        "fixed pooling-10x",

        "DRIP-4x",
        "DRIP-8x",
        "DRIP-10x",
    ]

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        True,
        which="major",
        alpha=0.18,
        linewidth=0.8,
    )

    ax.set_axisbelow(True)

    # --------------------------------------------------------
    # Plot curves
    # --------------------------------------------------------

    for model_name in model_order:

        group = (
            df[df["Model"] == model_name]
            .sort_values("SFT_Data_K")
        )

        if len(group) == 0:
            continue

        category = get_category(model_name)
        compression = get_compression(model_name)

        if category == "LLaVA":
            marker = markers["LLaVA"]
            linestyle = linestyles["LLaVA"]
            linewidth = 2.2
            markersize = 7.5
            alpha = 0.90

        else:
            marker = markers[compression]
            linestyle = linestyles[compression]

            if category == "DRIP":
                linewidth = 2.5
                markersize = 7.5
                alpha = 0.95
            else:
                linewidth = 1.9
                markersize = 7.0
                alpha = 0.78

        ax.plot(
            group["SFT_Data_K"],
            group[score_col],
            color=colors[category],
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            markersize=markersize,
            markeredgecolor="white",
            markeredgewidth=1.0,
            alpha=alpha,
            label=pretty_model_name(model_name),
            zorder=3 if category == "DRIP" else 2,
        )

    # --------------------------------------------------------
    # Full LLaVA reference
    # --------------------------------------------------------

    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.35,
        zorder=1,
    )

    # --------------------------------------------------------
    # X-axis
    # --------------------------------------------------------

    data_scales = np.array([
        0.25,
        0.50,
        0.75,
        1.00,
    ])

    x_ticks = (
        data_scales
        * FULL_SFT_SIZE_K
    )

    ax.set_xticks(x_ticks)

    ax.set_xticklabels([
        f"{x:.0f}K"
        if x.is_integer()
        else f"{x:.1f}K"
        for x in x_ticks
    ])

    ax.set_xlim(
        x_ticks[0] - 25,
        x_ticks[-1] + 25,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_title(
        title,
        pad=12,
        fontweight="bold",
    )

    ax.set_ylabel(ylabel)

    # --------------------------------------------------------
    # Clean spines
    # --------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# Combined figure
# ============================================================

def plot_combined_scaling(df):
    setup_plot_style()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.8, 5.2),
        sharex=True,
        sharey=True,
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    plot_scaling_panel(
        ax=axes[0],
        df=df,
        score_col="OverallScore",
        title="Overall Performance",
        ylabel="Average Relative Performance",
    )

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    plot_scaling_panel(
        ax=axes[1],
        df=df,
        score_col="OCRScore",
        title="OCR Performance",
        ylabel="",
    )

    # --------------------------------------------------------
    # Optional custom y limits
    # --------------------------------------------------------

    axes[0].set_ylim(
        0.835,
        1.015,
    )

    axes[1].set_ylim(
        0.70,
        1.015,
    )

    # --------------------------------------------------------
    # Shared x label
    # --------------------------------------------------------

    fig.supxlabel(
        "SFT Data Size",
        y=0.06,
        fontsize=13,
    )

    # --------------------------------------------------------
    # Shared legend
    #
    # Pull handles from first subplot.
    # --------------------------------------------------------

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        ncol=7,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        bbox_to_anchor=(0.5, -0.02),
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.tight_layout(
        rect=[
            0,
            0.10,
            1,
            1,
        ]
    )

    return fig, axes


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    df = pd.read_csv(INPUT_CSV)

    df = prepare_data(df)

    # --------------------------------------------------------
    # Print useful values
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SFT scaling scores")
    print("=" * 80)

    print(
        df[
            [
                "Model",
                "data_scale",
                "SFT_Data_K",
                "OverallScore",
                "OCRScore",
            ]
        ]
        .sort_values(
            [
                "Model",
                "data_scale",
            ]
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plot_combined_scaling(df)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PDF),
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_PNG,
        dpi=400,
        bbox_inches="tight",
    )

    print()
    print(f"Saved PDF -> {OUTPUT_PDF}")
    print(f"Saved PNG -> {OUTPUT_PNG}")

    plt.show()