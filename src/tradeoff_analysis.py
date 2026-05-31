import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

CSV_ID = "lora_7B_last"

df = pd.read_csv(f"results/{CSV_ID}.csv")



ocr_cols = ["TextVQA", "OCRBench", "OCRBenchv2", "DocVQA", "ChartQAPro"]
all_metric_cols = ["VQAv2", "SQA", "MME", "MM-Bench", "GQA", "MMMU", "TextVQA", "OCRBench", "OCRBenchv2", "DocVQA", "ChartQAPro", "POPE", "LLaVA-Wild", "MM-Vet"]
scaler = MinMaxScaler()
normalized = pd.DataFrame(
    scaler.fit_transform(df[all_metric_cols]),
    columns=all_metric_cols
)
df["OverallScore"] = normalized[all_metric_cols].mean(axis=1)
df["OCRScore"] = normalized[ocr_cols].mean(axis=1)

# =========================
# Plot 1: Overall Tradeoff
# =========================

plt.figure(figsize=(8, 6))

for _, row in df.iterrows():
    plt.scatter(row["TFLOPs"], row["OverallScore"], s=100)
    plt.annotate(
        row["Model"],
        (row["TFLOPs"], row["OverallScore"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

plt.xlabel("TFLOPs")
plt.ylabel("Overall Score (Min-Max Normalized)")
plt.title("Overall Performance vs Compute")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(f"results/{CSV_ID}_overall_tradeoff.png", dpi=300)
plt.show()

# =========================
# Plot 2: OCR Tradeoff
# =========================

plt.figure(figsize=(8, 6))

for _, row in df.iterrows():
    plt.scatter(row["TFLOPs"], row["OCRScore"], s=100)
    plt.annotate(
        row["Model"],
        (row["TFLOPs"], row["OCRScore"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

plt.xlabel("TFLOPs")
plt.ylabel("OCR Score (Min-Max Normalized)")
plt.title("OCR Performance vs Compute")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(f"results/{CSV_ID}_ocr_tradeoff.png", dpi=300)
plt.show()