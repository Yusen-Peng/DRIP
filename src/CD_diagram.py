import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import friedmanchisquare, studentized_range


# ============================================================
# Configuration
# ============================================================

CSV_ID = "full_7B_last"

INPUT_CSV = f"results/{CSV_ID}.csv"

OUTPUT_PDF = f"results/{CSV_ID}_cd_combined.pdf"

ALPHA = 0.05


# ============================================================
# Benchmarks
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


COMPRESSION_RATIOS = [
    "4x",
    "8x",
    "10x",
]


# ============================================================
# Methods
# ============================================================

METHODS = [
    "DRIP",
    "fixed pooling",
    "PruMerge",
    "PruneSID",
    "Perceiver",
]


DISPLAY_NAMES = {
    "DRIP": "DRIP",
    "fixed pooling": "Fixed Pooling",
    "PruMerge": "PruMerge",
    "PruneSID": "PruneSID",
    "Perceiver": "Perceiver",
}


# ============================================================
# Plot style
# ============================================================

def setup_plot_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 13,

        "axes.titlesize": 15,
        "axes.labelsize": 13,

        "xtick.labelsize": 11,
        "ytick.labelsize": 11,

        "axes.linewidth": 1.0,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# ============================================================
# Parse model names
# ============================================================

def parse_model_name(model_name):
    """
    Convert:

        DRIP-4x
        fixed pooling-8x
        PruMerge-10x

    into:

        ("DRIP", "4x")
        ("fixed pooling", "8x")
        ("PruMerge", "10x")

    Baseline models such as LLaVA-1.5-7B return:
        (None, None)
    """

    model_name = str(model_name).strip()

    match = re.match(
        r"^(.*)-(4x|8x|10x)$",
        model_name,
    )

    if match is None:
        return None, None

    method = match.group(1).strip()
    compression = match.group(2).strip()

    return method, compression


# ============================================================
# Construct benchmark × compression tasks
# ============================================================

def build_task_score_matrix(df, benchmarks):
    """
    Constructs:

        rows = benchmark × compression settings
        cols = methods

    For ALL_BENCHMARKS:
        14 × 3 = 42 tasks

    For OCR_BENCHMARKS:
         5 × 3 = 15 tasks

    Example:

        VQAv2 @ 4x

    compares:

        DRIP-4x
        fixed pooling-4x
        PruMerge-4x
        PruneSID-4x
        Perceiver-4x
    """

    df = df.copy()

    parsed = df["Model"].apply(parse_model_name)

    df["Method"] = parsed.apply(
        lambda x: x[0]
    )

    df["Compression"] = parsed.apply(
        lambda x: x[1]
    )

    # Remove LLaVA baseline / anything without compression suffix.
    df = df[
        df["Method"].notna()
    ].copy()

    rows = []

    for compression in COMPRESSION_RATIOS:

        compression_df = df[
            df["Compression"] == compression
        ]

        for benchmark in benchmarks:

            row = {
                "Task": f"{benchmark} @ {compression}",
                "Benchmark": benchmark,
                "Compression": compression,
            }

            for method in METHODS:

                method_row = compression_df[
                    compression_df["Method"] == method
                ]

                if len(method_row) != 1:
                    raise ValueError(
                        f"\nExpected exactly one row for:\n"
                        f"  method      = {method}\n"
                        f"  compression = {compression}\n\n"
                        f"Found {len(method_row)} rows.\n"
                    )

                value = method_row.iloc[0][benchmark]

                value = pd.to_numeric(
                    value,
                    errors="coerce",
                )

                if pd.isna(value):
                    raise ValueError(
                        f"Missing/non-numeric value for "
                        f"{method}, "
                        f"{compression}, "
                        f"{benchmark}"
                    )

                row[method] = float(value)

            rows.append(row)

    task_scores = pd.DataFrame(rows)

    expected_num_tasks = (
        len(benchmarks)
        * len(COMPRESSION_RATIOS)
    )

    if len(task_scores) != expected_num_tasks:
        raise RuntimeError(
            f"Expected {expected_num_tasks} tasks, "
            f"but constructed {len(task_scores)}."
        )

    return task_scores


# ============================================================
# Rank methods per task
# ============================================================

def compute_task_ranks(task_scores):
    """
    Higher benchmark score = better.

    Therefore:

        best   -> rank 1
        worst  -> rank 5

    Ties receive average ranks.
    """

    task_ranks = task_scores[
        [
            "Task",
            "Benchmark",
            "Compression",
        ]
    ].copy()

    score_matrix = task_scores[METHODS]

    rank_matrix = score_matrix.rank(
        axis=1,
        ascending=False,
        method="average",
    )

    for method in METHODS:
        task_ranks[method] = rank_matrix[method]

    return task_ranks


# ============================================================
# Friedman + Nemenyi
# ============================================================

def run_statistics(
    task_ranks,
    alpha=0.05,
):
    rank_matrix = task_ranks[METHODS]

    N = len(rank_matrix)
    k = len(METHODS)

    # --------------------------------------------------------
    # Friedman omnibus test
    # --------------------------------------------------------

    friedman_stat, friedman_p = friedmanchisquare(
        *[
            rank_matrix[method].to_numpy()
            for method in METHODS
        ]
    )

    # --------------------------------------------------------
    # Average ranks
    # --------------------------------------------------------

    avg_ranks = (
        rank_matrix
        .mean(axis=0)
        .sort_values()
    )

    # --------------------------------------------------------
    # Nemenyi critical difference
    # --------------------------------------------------------
    #
    # CD =
    #
    # q_alpha *
    # sqrt(
    #     k(k+1)
    #     -------
    #      6N
    # )
    #
    # scipy gives the Studentized Range critical value.
    #
    # The conventional Nemenyi q is:
    #
    # q_studentized / sqrt(2)
    #
    # --------------------------------------------------------

    q_studentized = studentized_range.ppf(
        1.0 - alpha,
        k,
        np.inf,
    )

    q_nemenyi = (
        q_studentized
        / np.sqrt(2.0)
    )

    cd = (
        q_nemenyi
        * np.sqrt(
            k * (k + 1)
            / (6.0 * N)
        )
    )

    return {
        "N": N,
        "k": k,

        "friedman_stat": friedman_stat,
        "friedman_p": friedman_p,

        "q_studentized": q_studentized,
        "q_nemenyi": q_nemenyi,

        "cd": cd,

        "average_ranks": avg_ranks,
    }


# ============================================================
# Find non-significant groups
# ============================================================

def find_nonsignificant_groups(
    avg_ranks,
    cd,
):
    """
    Find maximal contiguous groups for which the difference
    between the leftmost and rightmost average ranks <= CD.

    These are drawn as thick horizontal lines.
    """

    avg_ranks = avg_ranks.sort_values()

    ranks = avg_ranks.to_numpy()

    candidate_groups = []

    n = len(ranks)

    # --------------------------------------------------------
    # Find all intervals satisfying CD
    # --------------------------------------------------------

    for start in range(n):

        for end in range(
            start + 1,
            n,
        ):

            difference = (
                ranks[end]
                - ranks[start]
            )

            if difference <= cd:
                candidate_groups.append(
                    (start, end)
                )

    # --------------------------------------------------------
    # Keep only maximal intervals
    # --------------------------------------------------------

    maximal_groups = []

    for group in candidate_groups:

        start, end = group

        contained = False

        for other in candidate_groups:

            if other == group:
                continue

            other_start, other_end = other

            if (
                other_start <= start
                and other_end >= end
                and (
                    other_start < start
                    or other_end > end
                )
            ):
                contained = True
                break

        if not contained:
            maximal_groups.append(group)

    # Remove duplicates while preserving order.
    maximal_groups = list(
        dict.fromkeys(maximal_groups)
    )

    return maximal_groups


# ============================================================
# Print statistics
# ============================================================

def print_statistics(
    name,
    stats,
):
    avg_ranks = stats["average_ranks"]
    cd = stats["cd"]

    print("\n")
    print("=" * 80)
    print(name.upper())
    print("=" * 80)

    print(
        f"\nNumber of tasks: {stats['N']}"
    )

    print(
        f"Number of methods: {stats['k']}"
    )

    print("\nAverage ranks:")

    for method, rank in avg_ranks.items():
        print(
            f"  "
            f"{DISPLAY_NAMES[method]:15s}"
            f" = {rank:.4f}"
        )

    print("\nFriedman test:")

    print(
        f"  chi-square = "
        f"{stats['friedman_stat']:.6f}"
    )

    print(
        f"  p-value    = "
        f"{stats['friedman_p']:.8g}"
    )

    print("\nNemenyi:")

    print(
        f"  q           = "
        f"{stats['q_nemenyi']:.6f}"
    )

    print(
        f"  CD          = "
        f"{cd:.6f}"
    )

    print("\nPairwise comparisons:")

    methods_sorted = list(
        avg_ranks.index
    )

    for i in range(len(methods_sorted)):

        for j in range(
            i + 1,
            len(methods_sorted),
        ):

            method_a = methods_sorted[i]
            method_b = methods_sorted[j]

            rank_a = avg_ranks[method_a]
            rank_b = avg_ranks[method_b]

            difference = abs(
                rank_a - rank_b
            )

            significant = (
                difference > cd
            )

            result = (
                "SIGNIFICANT"
                if significant
                else "not significant"
            )

            print(
                f"  "
                f"{DISPLAY_NAMES[method_a]:15s}"
                f" vs "
                f"{DISPLAY_NAMES[method_b]:15s}"
                f" | Δrank = {difference:.3f}"
                f" | {result}"
            )


# ============================================================
# Draw one CD diagram into an existing axis
# ============================================================

def plot_cd_diagram(
    ax,
    avg_ranks,
    cd,
    title,
    num_tasks,
):
    """
    Draw a Critical Difference diagram into the provided axis.

    This function does NOT create or save a figure.
    """

    sorted_ranks = (
        avg_ranks
        .sort_values()
    )

    groups = find_nonsignificant_groups(
        sorted_ranks,
        cd,
    )

    k = len(sorted_ranks)

    min_rank = 1
    max_rank = k

    axis_y = 1.0

    # --------------------------------------------------------
    # How many labels on each side
    # --------------------------------------------------------

    left_count = int(
        np.ceil(k / 2)
    )

    sorted_items = list(
        sorted_ranks.items()
    )

    left_items = (
        sorted_items[:left_count]
    )

    right_items = (
        sorted_items[left_count:]
    )

    # --------------------------------------------------------
    # Main rank axis
    # --------------------------------------------------------

    ax.plot(
        [min_rank, max_rank],
        [axis_y, axis_y],
        color="black",
        linewidth=1.2,
        zorder=2,
    )

    # --------------------------------------------------------
    # Rank ticks
    # --------------------------------------------------------

    for rank in range(
        min_rank,
        max_rank + 1,
    ):

        ax.plot(
            [rank, rank],
            [
                axis_y - 0.035,
                axis_y + 0.035,
            ],
            color="black",
            linewidth=1.0,
        )

        ax.text(
            rank,
            axis_y + 0.08,
            str(rank),
            ha="center",
            va="bottom",
            fontsize=11,
        )

    # --------------------------------------------------------
    # Axis description
    # --------------------------------------------------------

    ax.text(
        (
            min_rank
            + max_rank
        ) / 2,
        axis_y + 0.22,
        "Average Rank (lower is better)",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # Label positioning
    # --------------------------------------------------------

    label_vertical_gap = 0.30

    # --------------------------------------------------------
    # Left-side labels
    # --------------------------------------------------------

    for i, (
        method,
        rank,
    ) in enumerate(left_items):

        y = (
            axis_y
            - 0.32
            - i * label_vertical_gap
        )

        label_x = (
            min_rank - 0.55
        )

        # vertical connector
        ax.plot(
            [rank, rank],
            [axis_y, y],
            color="black",
            linewidth=0.8,
        )

        # horizontal connector
        ax.plot(
            [
                label_x + 0.06,
                rank,
            ],
            [y, y],
            color="black",
            linewidth=0.8,
        )

        label = DISPLAY_NAMES.get(
            method,
            method,
        )

        ax.text(
            label_x,
            y,
            f"{label}  ({rank:.2f})",
            ha="right",
            va="center",
            fontsize=11.5,
            fontweight=(
                "bold"
                if method == "DRIP"
                else "normal"
            ),
        )

    # --------------------------------------------------------
    # Right-side labels
    # --------------------------------------------------------

    for i, (
        method,
        rank,
    ) in enumerate(right_items):

        y = (
            axis_y
            - 0.32
            - i * label_vertical_gap
        )

        label_x = (
            max_rank + 0.55
        )

        # vertical connector
        ax.plot(
            [rank, rank],
            [axis_y, y],
            color="black",
            linewidth=0.8,
        )

        # horizontal connector
        ax.plot(
            [
                rank,
                label_x - 0.06,
            ],
            [y, y],
            color="black",
            linewidth=0.8,
        )

        label = DISPLAY_NAMES.get(
            method,
            method,
        )

        ax.text(
            label_x,
            y,
            f"({rank:.2f})  {label}",
            ha="left",
            va="center",
            fontsize=11.5,
            fontweight=(
                "bold"
                if method == "DRIP"
                else "normal"
            ),
        )

    # --------------------------------------------------------
    # Non-significance bars
    # --------------------------------------------------------

    ranks = (
        sorted_ranks
        .to_numpy()
    )

    group_start_y = (
        axis_y + 0.40
    )

    group_gap = 0.09

    for i, (
        start,
        end,
    ) in enumerate(groups):

        y = (
            group_start_y
            + i * group_gap
        )

        x1 = ranks[start]
        x2 = ranks[end]

        ax.plot(
            [x1, x2],
            [y, y],
            color="black",
            linewidth=4.0,
            solid_capstyle="butt",
            zorder=3,
        )

    # --------------------------------------------------------
    # Critical Difference bar
    # --------------------------------------------------------

    cd_y = (
        group_start_y
        + max(
            len(groups),
            1,
        ) * group_gap
        + 0.20
    )

    cd_x1 = min_rank
    cd_x2 = min_rank + cd

    ax.plot(
        [cd_x1, cd_x2],
        [cd_y, cd_y],
        color="black",
        linewidth=1.8,
    )

    # Left endpoint
    ax.plot(
        [cd_x1, cd_x1],
        [
            cd_y - 0.035,
            cd_y + 0.035,
        ],
        color="black",
        linewidth=1.1,
    )

    # Right endpoint
    ax.plot(
        [cd_x2, cd_x2],
        [
            cd_y - 0.035,
            cd_y + 0.035,
        ],
        color="black",
        linewidth=1.1,
    )

    ax.text(
        (
            cd_x1
            + cd_x2
        ) / 2,
        cd_y + 0.06,
        f"CD = {cd:.2f}",
        ha="center",
        va="bottom",
        fontsize=10.5,
    )

    # --------------------------------------------------------
    # Panel title
    # --------------------------------------------------------

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=12,
    )

    # Small task-count subtitle
    ax.text(
        0.5,
        1.0,
        f"{num_tasks} benchmark-compression tasks",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10
    )

    # --------------------------------------------------------
    # Layout bounds
    # --------------------------------------------------------

    max_rows = max(
        len(left_items),
        len(right_items),
    )

    bottom = (
        axis_y
        - 0.42
        - max_rows
        * label_vertical_gap
    )

    top = (
        cd_y + 0.24
    )

    ax.set_xlim(
        min_rank - 2.15,
        max_rank + 2.15,
    )

    ax.set_ylim(
        bottom,
        top,
    )

    ax.axis("off")


# ============================================================
# Main
# ============================================================

def main():

    setup_plot_style()

    os.makedirs(
        os.path.dirname(OUTPUT_PDF),
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_CSV
    )

    print(
        f"\nLoaded: {INPUT_CSV}"
    )

    print(
        f"Rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    missing_columns = [
        benchmark
        for benchmark in ALL_BENCHMARKS
        if benchmark not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing benchmark columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    # ========================================================
    # OVERALL
    # ========================================================

    overall_scores = (
        build_task_score_matrix(
            df,
            ALL_BENCHMARKS,
        )
    )

    overall_ranks = (
        compute_task_ranks(
            overall_scores
        )
    )

    overall_stats = (
        run_statistics(
            overall_ranks,
            alpha=ALPHA,
        )
    )

    # ========================================================
    # OCR
    # ========================================================

    ocr_scores = (
        build_task_score_matrix(
            df,
            OCR_BENCHMARKS,
        )
    )

    ocr_ranks = (
        compute_task_ranks(
            ocr_scores
        )
    )

    ocr_stats = (
        run_statistics(
            ocr_ranks,
            alpha=ALPHA,
        )
    )

    # --------------------------------------------------------
    # Print both statistical analyses
    # --------------------------------------------------------

    print_statistics(
        "Overall",
        overall_stats,
    )

    print_statistics(
        "OCR",
        ocr_stats,
    )

    # ========================================================
    # Combined figure
    # ========================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16.5, 5.2),
    )

    # --------------------------------------------------------
    # Overall panel
    # --------------------------------------------------------

    plot_cd_diagram(
        ax=axes[0],
        avg_ranks=overall_stats[
            "average_ranks"
        ],
        cd=overall_stats["cd"],
        title="(a) Overall",
        num_tasks=overall_stats["N"],
    )

    # --------------------------------------------------------
    # OCR panel
    # --------------------------------------------------------

    plot_cd_diagram(
        ax=axes[1],
        avg_ranks=ocr_stats[
            "average_ranks"
        ],
        cd=ocr_stats["cd"],
        title="(b) OCR",
        num_tasks=ocr_stats["N"],
    )

    # --------------------------------------------------------
    # Spacing
    # --------------------------------------------------------

    plt.subplots_adjust(
        left=0.03,
        right=0.97,
        top=0.90,
        bottom=0.06,
        wspace=0.12,
    )

    # --------------------------------------------------------
    # Save ONE PDF
    # --------------------------------------------------------

    fig.savefig(
        OUTPUT_PDF,
        bbox_inches="tight",
    )

    fig.savefig(
        OUTPUT_PDF.replace(".pdf", ".png"),
        bbox_inches="tight",
        dpi=300,
    )

    print("\n" + "=" * 80)

    print(
        f"Saved combined CD diagram to:\n"
        f"{OUTPUT_PDF}"
    )

    print("=" * 80)

    plt.show()


if __name__ == "__main__":
    main()