import argparse
import json
import re
import string
from collections import Counter

def normalize_answer(text):
    """Normalize answer before comparison/tokenization."""
    text = str(text).lower().strip()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Normalize whitespace
    text = " ".join(text.split())
    return text


def token_f1(prediction, ground_truth):
    """Token-overlap F1 for one question."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        item["question_id"]: item
        for item in data
    }


def load_predictions(path):
    predictions = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            predictions[item["question_id"]] = item

    return predictions


def evaluate(gt_path, pred_path):
    ground_truth = load_ground_truth(gt_path)
    predictions = load_predictions(pred_path)

    binary_correct = 0
    binary_total = 0

    open_f1_scores = []

    missing = 0

    for qid, gt_item in ground_truth.items():

        if qid not in predictions:
            missing += 1
            continue

        gt = gt_item["answer"]
        pred = predictions[qid]["text"]

        gt_norm = normalize_answer(gt)
        pred_norm = normalize_answer(pred)

        # -------------------------
        # Binary: Yes / No Accuracy
        # -------------------------
        if gt_norm in {"yes", "no"}:
            binary_total += 1

            if pred_norm == gt_norm:
                binary_correct += 1

        # -------------------------
        # Open-ended: Token F1
        # -------------------------
        else:
            f1 = token_f1(pred, gt)
            open_f1_scores.append(f1)

    yes_no_accuracy = (
        binary_correct / binary_total
        if binary_total > 0 else 0.0
    )

    macro_f1 = (
        sum(open_f1_scores) / len(open_f1_scores)
        if open_f1_scores else 0.0
    )

    print("=" * 55)
    print("PathVQA Evaluation")
    print("=" * 55)

    print(f"Ground truth       : {len(ground_truth)}")
    print(f"Predictions        : {len(predictions)}")
    print(f"Missing            : {missing}")

    print()

    print(f"Binary questions   : {binary_total}")
    print(f"Open questions     : {len(open_f1_scores)}")

    print()

    print(f"Yes/No Accuracy    : {yes_no_accuracy * 100:.2f}")
    print(f"Open Macro F1      : {macro_f1 * 100:.2f}")

    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ground-truth",
        required=True,
    )

    parser.add_argument(
        "--predictions",
        required=True,
    )

    args = parser.parse_args()

    evaluate(
        args.ground_truth,
        args.predictions,
    )