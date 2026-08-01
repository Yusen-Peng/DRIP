import argparse
import json
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_docvqa_correctness(eval_file):
    data = load_json(eval_file)

    correctness = {}
    scores = {}

    for qid, item in data["details"].items():
        score = float(item["anls"])

        correctness[str(qid)] = int(score > 0.0)
        scores[str(qid)] = score

    return correctness, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-file", required=True)
    parser.add_argument("--method-file", required=True)
    parser.add_argument("--exact", action="store_true")
    args = parser.parse_args()

    baseline, baseline_scores = load_docvqa_correctness(args.baseline_file)
    method, method_scores = load_docvqa_correctness(args.method_file)

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

    baseline_anls = np.mean([baseline_scores[qid] for qid in common_ids]) * 100
    method_anls = np.mean([method_scores[qid] for qid in common_ids]) * 100

    baseline_bin = (a + b) / len(common_ids) * 100
    method_bin = (a + c) / len(common_ids) * 100

    print("=" * 70)
    print("DocVQA McNemar Test")
    print("=" * 70)
    print(f"N: {len(common_ids)}")
    print("Binarization: ANLS > 0.0")
    print()

    print("Contingency table")
    print()
    print("                         Method correct    Method wrong")
    print(f"Baseline correct      {a:>8}            {b:>8}")
    print(f"Baseline wrong        {c:>8}            {d:>8}")
    print()

    print(f"Baseline ANLS: {baseline_anls:.2f}")
    print(f"Method ANLS:   {method_anls:.2f}")
    print(f"ANLS delta:    {method_anls - baseline_anls:+.2f}")
    print()

    print(f"Baseline Accuracy@NL<0.5: {baseline_bin:.2f}")
    print(f"Method Accuracy@NL<0.5:   {method_bin:.2f}")
    print(f"Binary delta:             {method_bin - baseline_bin:+.2f}")
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