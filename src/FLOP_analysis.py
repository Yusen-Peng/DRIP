import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv("src/FLOP_data.csv")

    # Plot
    plt.figure(figsize=(8, 6))

    for model_name, group in df.groupby("model"):
        x = group["patch_size"].to_numpy()
        y = group["GFLOPS"].to_numpy()
        plt.plot(x, y, marker='o', label=model_name)

    plt.xlabel("Patch Size")
    plt.ylabel("GFLOPs")
    plt.title("FLOPs vs. Patch Size for ViT and DTP-ViT Models")
    plt.xscale('log', base=2)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("src/FLOP_analysis_plot.png")