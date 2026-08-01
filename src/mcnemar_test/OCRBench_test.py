import argparse
import json
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def compute_ocrbench_result(item):
    """
    Match model_vqa_ocrbench.py evaluation logic.

    Correct if any GT answer is substring of prediction.
    HME100k removes all spaces; others lowercase and replace newlines.
    """

    dataset_name = item["dataset_name"]
    answers = item["answers"]

    if item.get("predict", 0) == 0:
        return 0

    predict = item["predict"]

    if dataset_name == "HME100k":
        if isinstance(answers, list):
            for ans in answers:
                answer = ans.strip().replace("\n", " ").replace(" ", "")
                pred = predict.strip().replace("\n", " ").replace(" ", "")
                if answer in pred:
                    return 1
        else:
            answer = answers.strip().replace("\n", " ").replace(" ", "")
            pred = predict.strip().replace("\n", " ").replace(" ", "")
            if answer in pred:
                return 1
    else:
        if isinstance(answers, list):
            for ans in answers:
                answer = ans.lower().strip().replace("\n", " ")
                pred = predict.lower().strip().replace("\n", " ")
                if answer in pred:
                    return 1
        else:
            answer = answers.lower().strip().replace("\n", " ")
            pred = predict.lower().strip().replace("\n", " ")
            if answer in pred:
                return 1

    return 0


def make_qid(item, idx):
    """
    OCRBench JSON may not have a universal question_id.
    Use stable identifying fields.
    """

    image_path = str(item.get("image_path", ""))
    question = str(item.get("question", ""))
    dataset_name = str(item.get("dataset_name", ""))
    data_type = str(item.get("type", ""))

    return f"{idx}::{dataset_name}::{data_type}::{image_path}::{question}"


def load_ocrbench_correctness(result_file):
    data = load_json(result_file)

    correctness = {}

    for idx, item in enumerate(data):
        qid = make_qid(item, idx)
        correctness[qid] = compute_ocrbench_result(item)

    return correctness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-file", required=True)
    parser.add_argument("--method-file", required=True)
    parser.add_argument("--exact", action="store_true")
    args = parser.parse_args()

    baseline = load_ocrbench_correctness(args.baseline_file)
    method = load_ocrbench_correctness(args.method_file)

    common_ids = sorted(set(baseline.keys()) & set(method.keys()))

    if len(common_ids) == 0:
        raise ValueError("No overlapping question IDs found.")

    if set(baseline.keys()) != set(method.keys()):
        print("[Warning] Question ID sets are not identical.")
        print(f"Baseline-only: {len(set(baseline.keys()) - set(method.keys()))}")
        print(f"Method-only: {len(set(method.keys()) - set(baseline.keys()))}")
        raise ValueError("Question ID sets are not identical.")

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

    table = np.array([[a, b], [c, d]])

    if b + c == 0:
        raise ValueError("b + c = 0, McNemar test is undefined.")

    result = mcnemar(
        table,
        exact=args.exact,
        correction=not args.exact,
    )

    baseline_score = a + b
    method_score = a + c

    baseline_acc = baseline_score / len(common_ids) * 100
    method_acc = method_score / len(common_ids) * 100

    print("=" * 70)
    print("McNemar Test for OCRBench")
    print("=" * 70)
    print(f"N:        {len(common_ids)}")
    print()
    print("Contingency table:")
    print()
    print("                         Method correct    Method wrong")
    print(f"Baseline correct      {a:>8}            {b:>8}")
    print(f"Baseline wrong        {c:>8}            {d:>8}")
    print()
    print(f"Baseline score:    {baseline_score}")
    print(f"Method score:      {method_score}")
    print(f"Score delta:       {method_score - baseline_score:+d}")
    print()
    print(f"Baseline accuracy: {baseline_acc:.2f}")
    print(f"Method accuracy:   {method_acc:.2f}")
    print(f"Delta:             {method_acc - baseline_acc:+.2f}")
    print()
    print(f"b = {b}  (baseline correct, method wrong)")
    print(f"c = {c}  (baseline wrong, method correct)")
    print()
    print(f"McNemar statistic: {result.statistic}")
    print(f"p-value: {result.pvalue:.8f}")

    if result.pvalue < 0.05:
        print("Significant: YES, p < 0.05")
    else:
        print("Significant: NO, p >= 0.05")


if __name__ == "__main__":
    main()