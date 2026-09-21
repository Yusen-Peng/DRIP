import pandas as pd

# Load results
df = pd.read_csv("results/OCR.csv")

# Baseline model
baseline_name = "SFT@65K"

# All benchmark columns
benchmarks = [col for col in df.columns if col != "Model"]

# Get baseline scores
baseline = (
    df.loc[df["Model"] == baseline_name, benchmarks]
    .iloc[0]
)

# Compute relative performance retention (%)
retention = df.copy()

for benchmark in benchmarks:
    retention[benchmark] = (
        df[benchmark] / baseline[benchmark] * 100
    )

# Average retention across all benchmarks
retention["Average"] = retention[benchmarks].mean(axis=1)

print(retention.round(2))