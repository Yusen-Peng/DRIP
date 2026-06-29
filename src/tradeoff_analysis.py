import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np


def get_category(model_name):
    if model_name.startswith("fixed pooling"):
        return "Fixed pooling"
    elif model_name.startswith("PruMerge"):
        return "PruMerge"
    elif model_name.startswith("DRIP"):
        return "DRIP"
    elif model_name.startswith("LLaVA"):
        return "LLaVA"
    elif model_name.startswith("PruneSID"):
        return "PruneSID"
    return "Other"


def pretty_model_name(name):
    if name == "LLaVA-1.5-7B":
        return "LLaVA"
    if "-" in name and "DRIP" in name:
        return name.split("-")[-1]   # "4x", "8x", "10x"
    return ""



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


def plot_tradeoff(ax, df, score_col, ylabel, title):
    setup_plot_style()

    colors = {
        "LLaVA": "#6E6E6E",
        "Fixed pooling": "#F28E2B",
        "PruMerge": "#59A14F",
        "PruneSID": "#4E79A7",
        "DRIP": "#E15759",
    }

    markers = {
        "LLaVA": "o",
        "Fixed pooling": "s",
        "PruMerge": "^",
        "PruneSID": "P",
        "DRIP": "D",
    }

    df = df.copy()
    df["Category"] = df["Model"].apply(get_category)

    baseline_tflops = df.loc[df["Model"] == "LLaVA-1.5-7B", "TFLOPs"].iloc[0]
    df["Speedup"] = baseline_tflops / df["TFLOPs"]
    
    # safety
    df["TFLOPs"] = pd.to_numeric(df["TFLOPs"], errors="coerce")
    df["OverallScore"] = pd.to_numeric(df["OverallScore"], errors="coerce")
    df["OCRScore"] = pd.to_numeric(df["OCRScore"], errors="coerce")


    # Light background grid
    ax.grid(True, which="major", alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)

    # Plot category lines
    plot_order = ["LLaVA", "PruMerge", "PruneSID", "Fixed pooling", "DRIP"]

    for category in plot_order:
        group = df[df["Category"] == category].sort_values("Speedup")
        if len(group) == 0:
            continue

        ax.plot(
            group["Speedup"],
            group[score_col],
            color=colors[category],
            linewidth=2.4 if category == "DRIP" else 1.8,
            alpha=0.95 if category == "DRIP" else 0.75,
            zorder=2,
        )

        ax.scatter(
            group["Speedup"],
            group[score_col],
            s=80,
            color=colors[category],
            marker=markers[category],
            edgecolor="white",
            linewidth=1.2,
            alpha=0.85,   # <-- add this
            zorder=3,
        )

    # Baseline horizontal reference
    ax.axhline(
        1.0,
        color="black",
        linewidth=1.0,
        linestyle="--",
        alpha=0.35,
        zorder=1,
    )

    # Annotate only points, but cleaner
    label_offsets = {
        "LLaVA-1.5-7B": (8, -8),
        "DRIP-4x": (-16, 10),
        "DRIP-8x": (-16, 10),
        "DRIP-10x": (-16, 10),
        "fixed pooling-4x": (-30, -14),
        "fixed pooling-8x": (8, -12),
        "fixed pooling-10x": (8, -12),
        "PruMerge-4x": (-28, -16),
        "PruMerge-8x": (8, -12),
        "PruMerge-10x": (8, -12),
        "PruneSID-4x": (-28, -16),
        "PruneSID-8x": (8, -12),
        "PruneSID-10x": (8, -12),
    }

    for _, row in df.iterrows():
        name = row["Model"]
        dx, dy = label_offsets.get(name, (6, 6))

        ax.annotate(
            pretty_model_name(name),
            xy=(row["Speedup"], row[score_col]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            color="#222222",
            ha="left",
            va="center",
        )

    drip = df[df["Category"] == "DRIP"].sort_values("Speedup")
    x = drip["Speedup"].astype(float).to_numpy()
    y = drip[score_col].astype(float).to_numpy()
    ax.fill_between(
        x,
        y,
        0.72,
        color=colors["DRIP"],
        alpha=0.045,
        zorder=0,
    )

    ax.set_title(title, pad=12, fontweight="bold")
    # ax.set_xlabel("TFLOP Speedup over LLaVA-1.5-7B")
    ax.set_ylabel(ylabel)

    ax.set_xlim(0.85, 4.75)
    ax.set_ylim(0.72, 1.015)

    # Cleaner legend
    handles = [
        Line2D(
            [0], [0],
            color=colors[c],
            marker=markers[c],
            linewidth=2.2,
            markersize=7,
            markeredgecolor="white",
            label=c,
        )
        for c in plot_order
    ]

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # fig.tight_layout()
    return handles


if __name__ == "__main__":
    CSV_ID = "full_7B_last"

    df = pd.read_csv(f"results/{CSV_ID}.csv")

    ocr_cols = [
        "TextVQA",
        "OCRBench",
        "OCRBenchv2",
        "DocVQA",
        "ChartQAPro",
    ]

    all_metric_cols = [
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

    baseline_row = df[df["Model"] == "LLaVA-1.5-7B"].iloc[0]

    relative = df[all_metric_cols].div(baseline_row[all_metric_cols], axis=1)

    df["OverallScore"] = relative[all_metric_cols].mean(axis=1)
    df["OCRScore"] = relative[ocr_cols].mean(axis=1)

    setup_plot_style()

    fig, axes = plt.subplots(

        1,

        2,

        figsize=(13.8, 5.2),

        sharey=True,

    )

    handles = plot_tradeoff(

        ax=axes[0],

        df=df,

        score_col="OverallScore",

        ylabel="Average Relative Performance",

        title="Overall Performance",

    )

    plot_tradeoff(

        ax=axes[1],

        df=df,

        score_col="OCRScore",

        ylabel="",

        title="OCR Performance",

    )

    fig.legend(

        handles=handles,

        loc="lower center",

        ncol=5,

        frameon=True,

        fancybox=True,

        framealpha=0.95,

        edgecolor="#DDDDDD",

        bbox_to_anchor=(0.5, -0.02),

    )

    fig.supxlabel("TFLOP Speedup over LLaVA-1.5-7B", y=0.05)

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    fig.savefig(

        f"results/{CSV_ID}_tradeoff_combined.pdf",

        bbox_inches="tight",

    )

    fig.savefig(

        f"results/{CSV_ID}_tradeoff_combined.png",

        dpi=400,

        bbox_inches="tight",

    )

    plt.show()
