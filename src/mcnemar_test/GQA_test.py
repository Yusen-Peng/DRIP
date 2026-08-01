import argparse
import json
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_gqa_predictions(path):
    """
    Loads converted GQA prediction JSON.
    Expected format:
    [
        {"questionId": "...", "prediction": "..."},
        ...
    ]
    """
    preds = load_json(path)
    pred_map = {}
    for item in preds:
        qid = str(item["questionId"])
        pred = str(item["prediction"]).strip()
        pred_map[qid] = pred
    return pred_map




def load_gqa_correctness(questions_file, predictions_file):
    """
    Computes per-question correctness for GQA balanced questions only.

    questions_file:
        testdev_balanced_questions.json

    predictions_file:
        testdev_balanced_predictions.json-style file

    Returns:
        correctness: dict[str(question_id)] -> 0/1
    """

    questions = load_json(questions_file)
    predictions = load_gqa_predictions(predictions_file)

    correctness = {}

    for qid, question in questions.items():
        qid = str(qid)

        if not question.get("isBalanced", False):
            continue

        if qid not in predictions:
            raise ValueError(f"Missing prediction for question ID: {qid}")

        gold = str(question["answer"]).strip()
        pred = predictions[qid]

        correctness[qid] = int(pred == gold)

    return correctness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-file", required=True)
    parser.add_argument("--baseline-file", required=True)
    parser.add_argument("--method-file", required=True)
    parser.add_argument("--exact", action="store_true")
    args = parser.parse_args()

    baseline = load_gqa_correctness(
        questions_file=args.questions_file,
        predictions_file=args.baseline_file,
    )

    method = load_gqa_correctness(
        questions_file=args.questions_file,
        predictions_file=args.method_file,
    )

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
    print("McNemar Test for GQA Overall Accuracy")
    print("=" * 70)
    print(f"N:        {len(common_ids)}")
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

