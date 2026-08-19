import os

import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

BENCHMARKS = [
    "TextVQA",
    "OCRBench",
    "OCRBenchv2",
    "DocVQA",
    "ChartQAPro",
]

COMPRESSION_RATES = [
    0.25,   # 4x
    0.125,  # 8x
    0.1,    # 10x
]

METRICS = [
    # "iou",
    "text_coverage",
]

BASE_DIR = "/users/PAS2912/yusenpeng/DRIP/src/example_analysis"

def run_paired_ttest(csv_path, benchmark, compression_rate, metric):
    df = pd.read_csv(csv_path)

    # NOTE: only use images where CRAFT detected text
    df = df[df["num_craft_boxes"] > 0].copy()

    # Pair DRIP and Fixed results from the same image.
    paired = (
        df.pivot(
            index="image",
            columns="method",
            values=metric,
        )
        .dropna(
            subset=[
                "DRIP",
                "Fixed",
            ]
        )
    )

    if len(paired) == 0:
        print(
            f"WARNING: no valid pairs for "
            f"{benchmark}, {compression_rate}, {metric}"
        )
        return None

    drip = paired["DRIP"]
    fixed = paired["Fixed"]
    diff = drip - fixed
    t_stat, p_value = ttest_rel(drip, fixed)

    return {
        "benchmark": benchmark,
        "compression_rate": compression_rate,
        "compression": f"{1 / compression_rate:.0f}x",
        "metric": metric,
        "n_pairs": len(paired),
        "drip_mean": drip.mean(),
        "drip_std": drip.std(),
        "fixed_mean": fixed.mean(),
        "fixed_std": fixed.std(),
        "mean_difference": diff.mean(),
        "t_statistic": t_stat,
        "p_value": p_value
    }

def main():

    rows = []

    for benchmark in BENCHMARKS:

        for compression_rate in COMPRESSION_RATES:

            csv_path = os.path.join(
                BASE_DIR,
                f"text_boundary_overlap_{benchmark}",
                f"craft_boundary_overlap_{compression_rate}.csv",
            )

            # print(
            #     f"\nProcessing:\n"
            #     f"  {csv_path}"
            # )

            for metric in METRICS:

                result = run_paired_ttest(
                    csv_path=csv_path,
                    benchmark=benchmark,
                    compression_rate=compression_rate,
                    metric=metric,
                )

                if result is not None:
                    rows.append(result)

    results = pd.DataFrame(rows)
    results = results.sort_values(
        [
            "benchmark",
            "compression_rate",
            "metric",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    # mark significance based on p-value < 0.05
    results["significant"] = results["p_value"] < 0.05
    print("\n")
    print("Paired T-test Results")
    print("=" * 40)

    display_cols = [
        "benchmark",
        "compression",
        "metric",
        "n_pairs",
        # "drip_mean",
        # "fixed_mean",
        # "mean_difference",
        "t_statistic",
        "p_value",
        "significant"
    ]

    print(
        results[
            display_cols
        ].to_string(
            index=False,
        )
    )


if __name__ == "__main__":
    main()
