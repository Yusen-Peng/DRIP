import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def get_category(model_name):
    if model_name.startswith("fixed pooling"):
        return "Fixed pooling"
    elif model_name.startswith("PruMerge"):
        return "PruMerge"
    elif model_name.startswith("DRIP"):
        return "DRIP"
    elif model_name.startswith("LLaVA"):
        return "LLaVA"
    else:
        return "Other"


def plot_tradeoff(df, score_col, ylabel, title, save_path):
    category_colors = {
        "LLaVA": "tab:gray",
        "Fixed pooling": "tab:orange",
        "PruMerge": "tab:green",
        "DRIP": "tab:red",
        "Other": "tab:gray",
    }

    df = df.copy()
    df["Category"] = df["Model"].apply(get_category)
    baseline_tflops = df.loc[df["Model"] == "LLaVA-1.5-7B", "TFLOPs"].iloc[0]
    df["Speedup"] = baseline_tflops / df["TFLOPs"]

    plt.figure(figsize=(8, 6))

    # Connect dots within each category
    for category, group in df.groupby("Category"):
        group = group.sort_values("Speedup")

        plt.plot(
            group["Speedup"],
            group[score_col],
            color=category_colors.get(category, "tab:gray"),
            linewidth=2.0,
            marker="o",
            alpha=0.75,
            zorder=1,
            label=category
        )

    # Scatter + labels
    for _, row in df.iterrows():
        category = row["Category"]

        plt.scatter(
            row["Speedup"],
            row[score_col],
            s=100,
            color=category_colors.get(category, "tab:gray"),
            edgecolor="black",
            linewidth=0.5,
            zorder=2
        )

        plt.annotate(
            row["Model"],
            (row["Speedup"], row[score_col]),
            xytext=(-20, -10),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8
        )

    plt.xlabel("TFLOP Speedup over LLaVA-1.5-7B")
    plt.title(title.replace("Compute", "Speedup"))
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.xlim(0.8, 5.0)
    plt.ylim(0.72, 1.01)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":

    CSV_ID = "lora_7B_last"

    df = pd.read_csv(f"results/{CSV_ID}.csv")

    ocr_cols = [
        "TextVQA",
        "OCRBench",
        "OCRBenchv2",
        "DocVQA",
        "ChartQAPro"
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
        "MM-Vet"
    ]

    baseline_row = df[df["Model"] == "LLaVA-1.5-7B"].iloc[0]
    relative = df[all_metric_cols].div(baseline_row[all_metric_cols], axis=1)
    df["OverallScore"] = relative[all_metric_cols].mean(axis=1)
    df["OCRScore"] = relative[ocr_cols].mean(axis=1)

    plot_tradeoff(
        df=df,
        score_col="OverallScore",
        ylabel="Overall Relative Performance vs LLaVA-1.5-7B",
        title="Overall Performance vs Compute",
        save_path=f"results/{CSV_ID}_overall_tradeoff.png"
    )

    plot_tradeoff(
        df=df,
        score_col="OCRScore",
        ylabel="OCR Relative Performance vs LLaVA-1.5-7B",
        title="OCR Performance vs Compute",
        save_path=f"results/{CSV_ID}_ocr_tradeoff.png"
    )