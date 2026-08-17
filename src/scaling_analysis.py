import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D


# ============================================================
# Configuration
# ============================================================

CSV_ID = "scaling"
INPUT_CSV = f"results/{CSV_ID}.csv"

OUTPUT_PDF = f"results/{CSV_ID}_sft_scaling_grid.pdf"
OUTPUT_PNG = f"results/{CSV_ID}_sft_scaling_grid.png"

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

        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        "axes.linewidth": 1.0,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def add_data_efficiency_annotation(
    ax,
    df,
    compression,
    score_col,
    drip_scale,
    fixed_scale=1.0,
):
    """
    Add a horizontal data-efficiency annotation comparing:

        DRIP @ drip_scale
        vs.
        Fixed Pooling @ fixed_scale

    The horizontal arrow is drawn at the Fixed Pooling performance level.
    """

    drip_name = f"DRIP-{compression}"
    fixed_name = f"fixed pooling-{compression}"

    # --------------------------------------------------------
    # Get the two points
    # --------------------------------------------------------

    drip_row = df[
        (df["Model"] == drip_name)
        & np.isclose(df["data_scale"], drip_scale)
    ].iloc[0]

    fixed_row = df[
        (df["Model"] == fixed_name)
        & np.isclose(df["data_scale"], fixed_scale)
    ].iloc[0]

    drip_x = drip_row["SFT_Data_K"]
    drip_y = drip_row[score_col]

    fixed_x = fixed_row["SFT_Data_K"]
    fixed_y = fixed_row[score_col]

    # We want to visually show that DRIP @ less data ≈ Fixed @ full data
    y = fixed_y
    # Horizontal double-headed arrow
    ax.annotate(
        "",
        xy=(drip_x, y),
        xytext=(fixed_x, y),
        arrowprops=dict(
            arrowstyle="<->",
            color="#0BE5B6",
            linewidth=1.0,
            linestyle="--",   # dashed
        ),
        zorder=5,
    )

    # --------------------------------------------------------
    # Text annotation
    # --------------------------------------------------------

    midpoint = (drip_x + fixed_x) / 2

    drip_k = drip_x
    fixed_k = fixed_x

    ax.annotate(
        f"DRIP @ {drip_k:.1f}K >= Fixed @ {fixed_k:.0f}K",
        xy=(midpoint, y),
        xytext=(-15, -15),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7,
        fontweight="bold",
        color="#333333",
    )


# ============================================================
# Model utilities
# ============================================================

def normalize_model_name(name):
    name = str(name).strip()

    name = name.replace(
        "fixed-pooling",
        "fixed pooling",
    )

    name = name.replace(
        "Fixed pooling",
        "fixed pooling",
    )

    name = name.replace(
        "Fixed-pooling",
        "fixed pooling",
    )

    return name


# ============================================================
# Data processing
# ============================================================

def prepare_data(df):

    df = df.copy()

    # --------------------------------------------------------
    # Normalize names
    # --------------------------------------------------------

    df["Model"] = df["Model"].apply(
        normalize_model_name
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

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
    # Fully trained uncompressed LLaVA baseline
    # --------------------------------------------------------

    baseline_mask = (
        (df["Model"] == "LLaVA-1.5-7B")
        &
        np.isclose(
            df["data_scale"],
            1.0,
        )
    )

    baseline_rows = df.loc[
        baseline_mask
    ]

    if len(baseline_rows) != 1:

        raise ValueError(
            "Expected exactly one "
            "fully-trained LLaVA-1.5-7B row."
        )

    baseline_row = baseline_rows.iloc[0]

    # --------------------------------------------------------
    # Normalize benchmark performance
    # --------------------------------------------------------

    relative = (
        df[ALL_BENCHMARKS]
        .div(
            baseline_row[ALL_BENCHMARKS],
            axis=1,
        )
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    df["OverallScore"] = (
        relative[ALL_BENCHMARKS]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    df["OCRScore"] = (
        relative[OCR_BENCHMARKS]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    # --------------------------------------------------------
    # Real dataset size
    # --------------------------------------------------------

    df["SFT_Data_K"] = (
        df["data_scale"]
        * FULL_SFT_SIZE_K
    )

    return df


# ============================================================
# Plot one panel
# ============================================================

def plot_single_panel(
    ax,
    df,
    compression,
    score_col,
):

    # --------------------------------------------------------
    # Colors
    # --------------------------------------------------------

    colors = {
        "LLaVA": "#6E6E6E",
        "Fixed": "#F28E2B",
        "DRIP": "#E15759",
    }

    # --------------------------------------------------------
    # Models for this compression ratio
    # --------------------------------------------------------

    llava_name = "LLaVA-1.5-7B"
    fixed_name = f"fixed pooling-{compression}"
    drip_name = f"DRIP-{compression}"

    plot_models = [
        (
            llava_name,
            "LLaVA",
        ),
        (
            fixed_name,
            "Fixed",
        ),
        (
            drip_name,
            "DRIP",
        ),
    ]

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    for model_name, label in plot_models:

        group = (
            df[
                df["Model"]
                == model_name
            ]
            .sort_values(
                "SFT_Data_K"
            )
        )

        if len(group) == 0:

            print(
                f"Warning: missing {model_name}"
            )

            continue

        if label == "LLaVA":

            linewidth = 2.0
            markersize = 6.5
            alpha = 0.85

        elif label == "DRIP":

            linewidth = 2.4
            markersize = 7.0
            alpha = 0.95

        else:

            linewidth = 1.9
            markersize = 6.5
            alpha = 0.80

        ax.plot(
            group["SFT_Data_K"],
            group[score_col],

            color=colors[label],

            marker="o",

            linewidth=linewidth,
            markersize=markersize,

            markeredgecolor="white",
            markeredgewidth=0.9,

            alpha=alpha,

            label=label,

            zorder=3
            if label == "DRIP"
            else 2,
        )

    # --------------------------------------------------------
    # Reference line
    # --------------------------------------------------------

    ax.axhline(
        1.0,
        color="black",
        linewidth=1.0,
        linestyle="--",
        alpha=0.30,
        zorder=1,
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        True,
        which="major",
        alpha=0.16,
        linewidth=0.75,
    )

    ax.set_axisbelow(True)

    # --------------------------------------------------------
    # X ticks
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

    ax.set_xticks(
        x_ticks
    )

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
    # Clean spines
    # --------------------------------------------------------

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


# ============================================================
# 2 x 3 combined figure
# ============================================================

def plot_scaling_grid(df):

    setup_plot_style()

    compression_ratios = [
        "4x",
        "8x",
        "10x",
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14.5, 8.2),
        sharex=True,
        sharey="row",
    )

    # ========================================================
    # Top row: Overall
    # ========================================================

    for col_idx, compression in enumerate(
        compression_ratios
    ):

        ax = axes[
            0,
            col_idx,
        ]

        plot_single_panel(
            ax=ax,
            df=df,
            compression=compression,
            score_col="OverallScore",
        )

        ax.set_title(
            f"{compression} Compression",
            fontweight="bold",
            pad=10,
        )

    # ========================================================
    # Bottom row: OCR
    # ========================================================

    for col_idx, compression in enumerate(
        compression_ratios
    ):

        ax = axes[
            1,
            col_idx,
        ]

        plot_single_panel(
            ax=ax,
            df=df,
            compression=compression,
            score_col="OCRScore",
        )

    # ========================================================
    # Y-axis ranges
    # ========================================================

    for ax in axes[0]:

        ax.set_ylim(
            0.84,
            1.015,
        )

    for ax in axes[1]:

        ax.set_ylim(
            0.74,
            1.015,
        )

    # ========================================================
    # Row labels
    # ========================================================

    axes[
        0,
        0,
    ].set_ylabel(
        "Overall Relative Performance"
    )

    axes[
        1,
        0,
    ].set_ylabel(
        "OCR Relative Performance"
    )

    # ========================================================
    # X labels
    # ========================================================

    for ax in axes[1]:

        ax.set_xlabel(
            "SFT Data Size"
        )

    # ========================================================
    # Row titles on left side
    # Optional but nice for paper figure
    # ========================================================

    fig.text(
        0.012,
        0.72,
        "Overall",
        rotation=90,
        va="center",
        ha="center",
        fontsize=15,
        fontweight="bold",
    )

    fig.text(
        0.012,
        0.30,
        "OCR",
        rotation=90,
        va="center",
        ha="center",
        fontsize=15,
        fontweight="bold",
    )


    # ========================================================
    # Data-efficiency annotations
    # ========================================================

    for col_idx, compression in enumerate(compression_ratios):

        # ----------------------------------------------------
        # Overall:
        # DRIP @ 50% data (~332.5K)
        # vs Fixed @ 100% data (665K)
        # ----------------------------------------------------

        add_data_efficiency_annotation(
            ax=axes[0, col_idx],
            df=df,
            compression=compression,
            score_col="OverallScore",
            drip_scale=0.50,
            fixed_scale=1.00,
        )

        # ----------------------------------------------------
        # OCR:
        # DRIP @ 25% data (~166.2K)
        # vs Fixed @ 100% data (665K)
        # ----------------------------------------------------

        add_data_efficiency_annotation(
            ax=axes[1, col_idx],
            df=df,
            compression=compression,
            score_col="OCRScore",
            drip_scale=0.25,
            fixed_scale=1.00,
        )


    # ========================================================
    # Shared legend
    # ========================================================

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#6E6E6E",
            marker="o",
            linewidth=2.0,
            markersize=7,
            markeredgecolor="white",
            label="Uncompressed LLaVA",
        ),

        Line2D(
            [0],
            [0],
            color="#F28E2B",
            marker="o",
            linewidth=2.0,
            markersize=7,
            markeredgecolor="white",
            label="Fixed Pooling",
        ),

        Line2D(
            [0],
            [0],
            color="#E15759",
            marker="o",
            linewidth=2.4,
            markersize=7,
            markeredgecolor="white",
            label="DRIP",
        ),
    ]

    fig.legend(
        handles=legend_handles,

        loc="lower center",

        ncol=3,

        frameon=True,

        fancybox=True,

        framealpha=0.95,

        edgecolor="#DDDDDD",

        bbox_to_anchor=(
            0.5,
            -0.005,
        ),
    )

    # ========================================================
    # Spacing
    # ========================================================

    plt.tight_layout(
        rect=[
            0.035,
            0.07,
            1.0,
            1.0,
        ],

        w_pad=2.0,
        h_pad=2.2,
    )

    return fig, axes


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_CSV
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    df = prepare_data(
        df
    )

    # --------------------------------------------------------
    # Print scores
    # --------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "SFT scaling scores"
    )

    print(
        "=" * 80
    )

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

    fig, axes = plot_scaling_grid(
        df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            OUTPUT_PDF
        ),
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
    print(
        f"Saved PDF -> {OUTPUT_PDF}"
    )

    print(
        f"Saved PNG -> {OUTPUT_PNG}"
    )

    plt.show()