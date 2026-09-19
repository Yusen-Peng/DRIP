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
    elif model_name.startswith("DITTO"):
        return "DITTO"
    elif model_name.startswith("LLaVA"):
        return "LLaVA"
    elif model_name.startswith("PruneSID"):
        return "PruneSID"
    elif model_name.startswith("Perceiver"):
        return "Perceiver"
    return "Other"


def pretty_model_name(name):
    if name == "LLaVA-1.5-7B" or name == "LLaVA-Qwen2.5-14B":
        return "LLaVA"
    if "-" in name and "DITTO" in name:
        return name.split("-")[-1]   # "4x", "8x", "10x"
    return ""



# def setup_plot_style():
#     mpl.rcParams.update({
#         "font.family": "serif",
#         "font.size": 12,
#         "axes.titlesize": 15,
#         "axes.labelsize": 13,
#         "legend.fontsize": 11,
#         "xtick.labelsize": 11,
#         "ytick.labelsize": 11,
#         "axes.linewidth": 1.1,
#         "pdf.fonttype": 42,
#         "ps.fonttype": 42,
#     })

def setup_plot_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # cleaner conference-style axes
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444444",

        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })




def plot_tradeoff(ax, df, score_col, ylabel, title):
    setup_plot_style()

    # colors = {
    #     "LLaVA": "#6E6E6E",
    #     "Fixed pooling": "#F28E2B",
    #     "PruMerge": "#59A14F",
    #     "PruneSID": "#4E79A7",
    #     "DITTO": "#E15759",
    #     "Perceiver": "#B07AA1",
    # }

    colors = {
        "LLaVA": "#777777",
        "Fixed pooling": "#E69F00",
        "PruMerge": "#7A9E65",
        "PruneSID": "#6C8EBF",
        "DITTO": "#D94A4A",
        "Perceiver": "#9B7E9B",
    }


    markers = {
        "LLaVA": "o",
        "Fixed pooling": "s",
        "PruMerge": "^",
        "PruneSID": "P",
        "DITTO": "D",
        "Perceiver": "X",
    }

    df = df.copy()
    df["Category"] = df["Model"].apply(get_category)

    if 'qwen' in CSV_ID:
        baseline_tflops = df.loc[df["Model"] == "LLaVA-Qwen2.5-14B", "TFLOPs"].iloc[0]
    else:
        baseline_tflops = df.loc[df["Model"] == "LLaVA-1.5-7B", "TFLOPs"].iloc[0]
    
    df["Speedup"] = baseline_tflops / df["TFLOPs"]
    
    # safety
    df["TFLOPs"] = pd.to_numeric(df["TFLOPs"], errors="coerce")
    df["Coarse-grained QA Score"] = pd.to_numeric(df["Coarse-grained QA Score"], errors="coerce")
    df["Fine-grained OCR Score"] = pd.to_numeric(df["Fine-grained OCR Score"], errors="coerce")

    ax.grid(
        True,
        axis="both",
        which="major",
        color="#D0D0D0",
        linewidth=0.6,
        alpha=0.28,
    )
    ax.set_axisbelow(True)



    if 'siglip' in CSV_ID.lower():
        plot_order = ["LLaVA", "Fixed pooling", "Perceiver", "DITTO"]
        present_plot_order = ["LLaVA", "Fixed pooling", "Perceiver", "DITTO(Ours)"]
    else:
        # Plot category lines
        plot_order = ["LLaVA", "PruMerge", "PruneSID", "Fixed pooling", "Perceiver", "DITTO"]
        present_plot_order = ["LLaVA", "PruMerge", "PruneSID", "Fixed pooling", "Perceiver", "DITTO(Ours)"]

    # for category in plot_order:
    #     group = df[df["Category"] == category].sort_values("Speedup")
    #     if len(group) == 0:
    #         continue

    #     ax.plot(
    #         group["Speedup"],
    #         group[score_col],
    #         color=colors[category],
    #         linewidth=2.4 if category == "DITTO" else 1.8,
    #         alpha=0.95 if category == "DITTO" else 0.75,
    #         zorder=2,
    #     )

    #     ax.scatter(
    #         group["Speedup"],
    #         group[score_col],
    #         s=80,
    #         color=colors[category],
    #         marker=markers[category],
    #         edgecolor="white",
    #         linewidth=1.2,
    #         alpha=0.85,   # <-- add this
    #         zorder=3,
    #     )
    for category in plot_order:
        group = df[df["Category"] == category].sort_values("Speedup")
        if len(group) == 0:
            continue

        is_drip = category == "DITTO"
        is_llava = category == "LLaVA"

        ax.plot(
            group["Speedup"],
            group[score_col],
            color=colors[category],
            linewidth=2.6 if is_drip else 1.5,
            alpha=1.0 if is_drip else 0.65,
            zorder=4 if is_drip else 2,
        )

        ax.scatter(
            group["Speedup"],
            group[score_col],
            s=72 if is_drip else 55,
            color=colors[category],
            marker=markers[category],
            edgecolor="white",
            linewidth=0.8,
            alpha=1.0 if is_drip else 0.75,
            zorder=5 if is_drip else 3,
        )

    # Baseline horizontal reference
    # ax.axhline(
    #     1.0,
    #     color="black",
    #     linewidth=1.0,
    #     linestyle="--",
    #     alpha=0.35,
    #     zorder=1,
    # )
    ax.axhline(
        1.0,
        color="#777777",
        linewidth=0.9,
        linestyle=(0, (4, 3)),
        alpha=0.55,
        zorder=1,
    )


    # Annotate only points, but cleaner
    # label_offsets = {
    #     "LLaVA-1.5-7B": (8, -8),
    #     "LLaVA-Qwen2.5-14B": (8, -8),
    #     "DITTO-4x": (-16, 10),
    #     "DITTO-8x": (-16, 10),
    #     "DITTO-10x": (-16, 10),
    #     "fixed pooling-4x": (-30, -14),
    #     "fixed pooling-8x": (8, -12),
    #     "fixed pooling-10x": (8, -12),
    #     "PruMerge-4x": (-28, -16),
    #     "PruMerge-8x": (8, -12),
    #     "PruMerge-10x": (8, -12),
    #     "PruneSID-4x": (-28, -16),
    #     "PruneSID-8x": (8, -12),
    #     "PruneSID-10x": (8, -12),
    #     "Perceiver-4x": (-28, -16),
    #     "Perceiver-8x": (8, -12),
    #     "Perceiver-10x": (8, -12),
    # }

    # for _, row in df.iterrows():
    #     name = row["Model"]
    #     dx, dy = label_offsets.get(name, (6, 6))

    #     ax.annotate(
    #         pretty_model_name(name),
    #         xy=(row["Speedup"], row[score_col]),
    #         xytext=(dx, dy),
    #         textcoords="offset points",
    #         fontsize=9,
    #         color="#222222",
    #         ha="left",
    #         va="center",
    #     )

    label_offsets = {
        "LLaVA-1.5-7B": (7, -8),
        "LLaVA-Qwen2.5-14B": (7, -8),

        "DITTO-4x": (-15, 11),
        "DITTO-8x": (-14, 11),
        "DITTO-10x": (-12, 11),
    }

    for _, row in df.iterrows():
        name = row["Model"]

        if row["Category"] not in ["DITTO", "LLaVA"]:
            continue

        dx, dy = label_offsets.get(name, (6, 6))

        if row["Category"] == "DITTO":
            label = name.split("-")[-1]
            weight = "semibold"
        else:
            label = "LLaVA"
            weight = "normal"

        ax.annotate(
            label,
            xy=(row["Speedup"], row[score_col]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            fontweight=weight,
            color="#333333",
            ha="left",
            va="center",
            zorder=6,
        )

    ax.text(
        0.99,
        1.002,
        "uncompressed",
        transform=ax.get_yaxis_transform(),
        fontsize=8.5,
        color="#777777",
        ha="right",
        va="bottom",
    )

    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(
        PercentFormatter(xmax=1.0, decimals=0)
    )


    # drip = df[df["Category"] == "DITTO"].sort_values("Speedup")
    # x = drip["Speedup"].astype(float).to_numpy()
    # y = drip[score_col].astype(float).to_numpy()

    # lower_bound = 0.50 if 'qwen' in CSV_ID else 0.72

    # ax.fill_between(
    #     x,
    #     y,
    #     lower_bound,
    #     color=colors["DITTO"],
    #     alpha=0.045,
    #     zorder=0,
    # )

    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_ylabel(ylabel)

    if 'qwen' in CSV_ID:
        ax.set_xlim(0.85, 5.95)
        ax.set_ylim(0.50, 1.015)
    elif 'SigLIP2' in CSV_ID:
        ax.set_xlim(0.85, 4.75)
        ax.set_ylim(0.80, 1.015)
    else:
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
    # CSV_ID = "full_7B_second_to_last"
    # CSV_ID = "qwen14B_full_last"
    # CSV_ID = "lora_7B_last"
    # CSV_ID = "lora_7B_second_to_last"
    # CSV_ID = "SigLIP2_7B_last"


    df = pd.read_csv(f"results/{CSV_ID}.csv")

    ocr_cols = [
        "TextVQA",
        "OCRBench",
        "OCRBenchv2",
        "DocVQA",
        "ChartQAPro",
    ]

    general_metric_cols = [
        "VQAv2",
        "SQA",
        "MME",
        "MM-Bench",
        "GQA",
        "MMMU",
        "POPE",
        "LLaVA-Wild",
        "MM-Vet",
    ]
    
    if 'qwen' in CSV_ID:
        baseline_row = df[df["Model"] == "LLaVA-Qwen2.5-14B"].iloc[0]
    else:
        baseline_row = df[df["Model"] == "LLaVA-1.5-7B"].iloc[0]

    relative = df[general_metric_cols + ocr_cols].div(baseline_row[general_metric_cols + ocr_cols], axis=1)
    df["Coarse-grained QA Score"] = relative[general_metric_cols].mean(axis=1)
    print("Coarse-grained QA Score:")
    print("=" * 40)
    # print model + score
    for model, score in zip(df["Model"], df["Coarse-grained QA Score"]):
        print(f"{model}: {score}")
    df["Fine-grained OCR Score"] = relative[ocr_cols].mean(axis=1)
    print("=" * 40)
    print("Fine-grained OCR Score:")
    for model, score in zip(df["Model"], df["Fine-grained OCR Score"]):
        print(f"{model}: {score}")

    setup_plot_style()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 5.2),
        sharey=True,
    )

    handles = plot_tradeoff(
        ax=axes[0],
        df=df,
        score_col="Coarse-grained QA Score",
        ylabel="Average Relative Performance",
        title="Coarse-grained QA Performance",
    )

    plot_tradeoff(
        ax=axes[1],
        df=df,
        score_col="Fine-grained OCR Score",
        ylabel="",
        title="Fine-grained OCR Performance",
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.125),
        columnspacing=1.5,
        handletextpad=0.5,
    )

    if "qwen" in CSV_ID:
        xlabel = "TFLOP Speedup over LLaVA-Qwen2.5-14B"

    else:

        xlabel = "TFLOP Speedup over LLaVA-1.5-7B"

    fig.supxlabel(
        xlabel,
        y=0.07,
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0.12, 1, 1])

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
