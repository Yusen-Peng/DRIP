import argparse
import json
import os
import numpy as np
from collections import defaultdict
from statsmodels.stats.contingency_tables import mcnemar


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_pope_pred(text):
    # Match eval_pope.py exactly
    if text.find(".") != -1:
        text = text.split(".")[0]

    text = text.replace(",", "")
    words = text.split(" ")

    if "No" in words or "not" in words or "no" in words:
        return 0
    else:
        return 1


def load_labels(annotation_dir):
    """
    Returns:
        labels_by_category[category] = list[int]
    """

    labels_by_category = {}

    for file in sorted(os.listdir(annotation_dir)):
        assert file.startswith("coco_pope_")
        assert file.endswith(".json")

        category = file[10:-5]
        label_file = os.path.join(annotation_dir, file)

        labels = []

        with open(label_file, "r") as f:
            for line in f:
                label = json.loads(line)["label"]

                if label == "no":
                    labels.append(0)
                else:
                    labels.append(1)

        labels_by_category[category] = labels

    return labels_by_category


def load_pope_correctness(annotation_dir, question_file, result_file):
    questions = load_jsonl(question_file)
    questions = {
        q["question_id"]: q
        for q in questions
    }

    answers = load_jsonl(result_file)

    labels_by_category = load_labels(annotation_dir)

    answers_by_category = defaultdict(list)

    for ans in answers:
        qid = ans["question_id"]
        category = questions[qid]["category"]
        answers_by_category[category].append(ans)

    correctness = {}
    category_correct = defaultdict(list)

    for category, labels in labels_by_category.items():
        cur_answers = answers_by_category[category]

        if len(cur_answers) != len(labels):
            raise ValueError(
                f"Category {category}: #answers={len(cur_answers)} "
                f"but #labels={len(labels)}"
            )

        for idx, (ans, label) in enumerate(zip(cur_answers, labels)):
            qid = str(ans["question_id"])
            pred = parse_pope_pred(ans["text"])

            correct = int(pred == label)

            key = f"{category}::{idx}::{qid}"

            correctness[key] = correct
            category_correct[category].append(correct)

    return correctness, category_correct


def macro_average_accuracy(category_correct):
    """
    Match your reported metric:
    average accuracy over the 3 POPE splits.
    """

    accs = []

    for category in sorted(category_correct.keys()):
        vals = category_correct[category]
        accs.append(np.mean(vals))

    return np.mean(accs) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--baseline-file", required=True)
    parser.add_argument("--method-file", required=True)
    parser.add_argument("--exact", action="store_true")
    args = parser.parse_args()

    baseline, baseline_by_cat = load_pope_correctness(
        args.annotation_dir,
        args.question_file,
        args.baseline_file,
    )

    method, method_by_cat = load_pope_correctness(
        args.annotation_dir,
        args.question_file,
        args.method_file,
    )

    if set(baseline.keys()) != set(method.keys()):
        raise ValueError("Question IDs do not match.")

    common_ids = sorted(baseline.keys())

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
        else:
            d += 1

    if b + c == 0:
        raise ValueError("McNemar undefined because b+c=0.")

    table = np.array([[a, b], [c, d]])

    result = mcnemar(
        table,
        exact=args.exact,
        correction=not args.exact,
    )

    baseline_acc = macro_average_accuracy(baseline_by_cat)
    method_acc = macro_average_accuracy(method_by_cat)

    baseline_micro = (a + b) / len(common_ids) * 100
    method_micro = (a + c) / len(common_ids) * 100

    print("=" * 70)
    print("POPE McNemar Test")
    print("=" * 70)
    print(f"N: {len(common_ids)}")
    print("Reported metric: average accuracy over POPE categories")
    print()

    print("Contingency table")
    print()
    print("                         Method correct    Method wrong")
    print(f"Baseline correct      {a:>8}            {b:>8}")
    print(f"Baseline wrong        {c:>8}            {d:>8}")
    print()

    print(f"Baseline macro accuracy: {baseline_acc:.2f}")
    print(f"Method macro accuracy:   {method_acc:.2f}")
    print(f"Macro delta:             {method_acc - baseline_acc:+.2f}")
    print()

    print(f"Baseline micro accuracy: {baseline_micro:.2f}")
    print(f"Method micro accuracy:   {method_micro:.2f}")
    print(f"Micro delta:             {method_micro - baseline_micro:+.2f}")
    print()

    print("Category accuracies:")
    for category in sorted(baseline_by_cat.keys()):
        base_cat = np.mean(baseline_by_cat[category]) * 100
        method_cat = np.mean(method_by_cat[category]) * 100
        print(
            f"  {category:<12} "
            f"baseline={base_cat:.2f} "
            f"method={method_cat:.2f} "
            f"delta={method_cat - base_cat:+.2f}"
        )

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
