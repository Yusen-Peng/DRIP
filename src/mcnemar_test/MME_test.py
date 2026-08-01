import argparse
import os
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

EVAL_TYPE_DICT = {
    "Perception": [
        "existence", "count", "position", "color", "posters",
        "celebrity", "scene", "landmark", "artwork", "OCR"
    ],
    "Cognition": [
        "commonsense_reasoning", "numerical_calculation",
        "text_translation", "code_reasoning"
    ],
}


def parse_pred_ans(pred_ans):
    pred_ans = pred_ans.strip().lower()
    if pred_ans in ["yes", "no"]:
        return pred_ans
    prefix = pred_ans[:4]
    if "yes" in prefix:
        return "yes"
    elif "no" in prefix:
        return "no"
    else:
        return "other"

def load_mme_correctness(results_dir, tasks=None):
    """
    Load converted MME .txt files.

    Each line format:
        img_name \t question \t gt_ans \t pred_ans

    Returns:
        correctness: dict[qid] -> 0/1
    """
    correctness = {}
    if tasks is None:
        tasks = []
        for task_list in EVAL_TYPE_DICT.values():
            tasks.extend(task_list)

    for task_name in tasks:
        task_txt = os.path.join(results_dir, task_name + ".txt")

        if not os.path.exists(task_txt):
            print(f"[Warning] Missing task file: {task_txt}")
            continue

        with open(task_txt, "r") as f:
            lines = f.readlines()

        for line_idx, line in enumerate(lines):
            parts = line.rstrip("\n").split("\t")

            if len(parts) != 4:
                raise ValueError(
                    f"Bad line in {task_txt}:{line_idx + 1}\n"
                    f"Expected 4 tab-separated fields, got {len(parts)}:\n{line}"
                )

            img_name, question, gt_ans, pred_ans = parts

            gt_ans = gt_ans.strip().lower()
            pred_label = parse_pred_ans(pred_ans)

            if gt_ans not in ["yes", "no"]:
                raise ValueError(f"Invalid GT answer in {task_txt}: {gt_ans}")
            # Unique per-question ID.
            # Include task because same image/question could theoretically appear across tasks.
            qid = f"{task_name}::{img_name}::{question}"
            correctness[qid] = int(pred_label == gt_ans)
    return correctness

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--method-dir", required=True)
    parser.add_argument("--exact", action="store_true")

    parser.add_argument(
        "--eval-type",
        choices=["all", "Perception", "Cognition"],
        default="all",
        help="Run McNemar on all tasks, perception only, or cognition only.",
    )

    parser.add_argument(
        "--task",
        default=None,
        help="Optional single task name, e.g. existence, OCR, count.",
    )

    args = parser.parse_args()

    if args.task is not None:
        tasks = [args.task]
        title = args.task
    elif args.eval_type == "all":
        tasks = None
        title = "MME All"
    else:
        tasks = EVAL_TYPE_DICT[args.eval_type]
        title = f"MME {args.eval_type}"

    baseline = load_mme_correctness(args.baseline_dir, tasks=tasks)
    method = load_mme_correctness(args.method_dir, tasks=tasks)

    base_ids = set(baseline.keys())
    method_ids = set(method.keys())

    if base_ids != method_ids:
        print("[Warning] Question ID sets are not identical.")
        print(f"Baseline-only: {len(base_ids - method_ids)}")
        print(f"Method-only: {len(method_ids - base_ids)}")
        raise ValueError("Question ID sets are not identical.")

    common_ids = sorted(base_ids & method_ids)

    if len(common_ids) == 0:
        raise ValueError("No overlapping question IDs found.")

    a = b = c = d = 0

    for qid in common_ids:
        base_correct = baseline[qid]
        method_correct = method[qid]

        if base_correct == 1 and method_correct == 1:
            a += 1
        elif base_correct == 1 and method_correct == 0:
            b += 1
        elif base_correct == 0 and method_correct == 1:
            c += 1
        elif base_correct == 0 and method_correct == 0:
            d += 1

    table = np.array([
        [a, b],
        [c, d]
    ])

    result = mcnemar(
        table,
        exact=args.exact,
        correction=not args.exact,
    )

    baseline_acc = (a + b) / len(common_ids) * 100
    method_acc = (a + c) / len(common_ids) * 100

    print("=" * 70)
    print(f"McNemar Test for {title}")
    print("=" * 70)
    print(f"N: {len(common_ids)}")
    print()
    print("Contingency table:")
    print()
    print("                         Method correct    Method wrong")
    print(f"Baseline correct      {a:>8}            {b:>8}")
    print(f"Baseline wrong        {c:>8}            {d:>8}")
    print()
    print(f"Baseline accuracy: {baseline_acc:.2f}")
    print(f"Method accuracy:   {method_acc:.2f}")
    print(f"Delta: {method_acc - baseline_acc:+.2f}")
    print()
    print(f"b = {b}  (Baseline correct, Method wrong)")
    print(f"c = {c}  (Baseline wrong, Method correct)")
    print()
    print(f"McNemar statistic: {result.statistic}")
    print(f"p-value: {result.pvalue:.8f}")

    if result.pvalue < 0.05:
        print("Significant: YES, p < 0.05")
    else:
        print("Significant: NO, p >= 0.05")


if __name__ == "__main__":
    main()
