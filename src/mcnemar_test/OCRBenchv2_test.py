import argparse
import json
import numpy as np
from collections import defaultdict
from statsmodels.stats.contingency_tables import mcnemar


EN_SPLITS = {
    "text_recognition": {
        "text recognition en",
        "fine-grained text recognition en",
        "full-page OCR en",
    },
    "visual_text_understanding": {
        "document classification en",
        "cognition VQA en",
        "diagram QA en",
    },
    "knowledge_reasoning": {
        "reasoning VQA en",
        "science QA en",
        "APP agent en",
        "ASCII art classification en",
    },
}

CN_SPLITS = {
    "text_recognition": {
        "full-page OCR cn",
    },
    "visual_text_understanding": {
        "cognition VQA cn",
    },
    "knowledge_reasoning": {
        "reasoning VQA cn",
        "text translation cn",
    },
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def make_qid(item, idx):
    image = str(item.get("image", item.get("image_path", "")))
    question = str(item.get("question", ""))
    task_type = str(item["type"])
    return f"{idx}::{task_type}::{image}::{question}"


def get_bucket(task_type):

    for bucket, task_set in EN_SPLITS.items():

        if task_type in task_set:

            return bucket

    return None


def load_results(path, threshold):
    data = load_json(path)

    correctness = {}
    scores = {}
    bucket_scores = defaultdict(list)

    for idx, item in enumerate(data):

        if item.get("ignore") == "True":
            continue

        bucket = get_bucket(item["type"])
        if bucket is None:
            continue

        if "score" not in item:
            raise ValueError(
                f"Missing score field at index {idx}"
            )

        score = float(item["score"])

        qid = make_qid(item, idx)

        correctness[qid] = int(score > threshold)
        scores[qid] = score

        bucket_scores[bucket].append(score)

    return correctness, scores, bucket_scores

def compute_reported_score(bucket_scores):

    avgs = []

    for bucket in EN_SPLITS:

        vals = bucket_scores.get(bucket, [])

        if len(vals):

            avgs.append(np.mean(vals))

    return np.mean(avgs) * 100



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--baseline-file", required=True)
    parser.add_argument("--method-file", required=True)

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="correct if score > threshold",
    )

    parser.add_argument(
        "--exact",
        action="store_true",
    )

    args = parser.parse_args()

    baseline, baseline_scores, baseline_buckets = load_results(
        args.baseline_file,
        args.threshold,
    )

    method, method_scores, method_buckets = load_results(
        args.method_file,
        args.threshold,
    )

    if set(baseline.keys()) != set(method.keys()):
        raise ValueError("Question IDs do not match")

    common_ids = sorted(baseline.keys())

    a = b = c = d = 0

    for qid in common_ids:

        base_correct = baseline[qid]
        method_correct = method[qid]

        if base_correct and method_correct:
            a += 1
        elif base_correct and not method_correct:
            b += 1
        elif not base_correct and method_correct:
            c += 1
        else:
            d += 1

    table = np.array([
        [a, b],
        [c, d],
    ])

    result = mcnemar(
        table,
        exact=args.exact,
        correction=not args.exact,
    )

    baseline_score = compute_reported_score(

        baseline_buckets

    )

    method_score = compute_reported_score(

        method_buckets

    )
    print("=" * 70)
    print("OCRBench-v2 McNemar Test")
    print("=" * 70)

    print(f"N = {len(common_ids)}")
    print(f"Threshold = {args.threshold}")
    print()

    print("Contingency Table")
    print()
    print("                         Method correct    Method wrong")
    print(f"Baseline correct      {a:>8}            {b:>8}")
    print(f"Baseline wrong        {c:>8}            {d:>8}")
    print()


    print("Reported OCRBench-v2 Score")

    print()

    print(

        f"Baseline score: {baseline_score:.2f}"

    )

    print(

        f"Method score:   {method_score:.2f}"

    )

    print(

        f"Delta:          {method_score - baseline_score:+.2f}"

    )

    print()

    print(
        f"Baseline Binary Accuracy: {(a+b)/len(common_ids)*100:.2f}"
    )
    print(
        f"Method Binary Accuracy:   {(a+c)/len(common_ids)*100:.2f}"
    )
    print()

    print(f"b = {b}")
    print(f"c = {c}")
    print()

    print(f"McNemar statistic: {result.statistic}")
    print(f"p-value: {result.pvalue:.8f}")

    if result.pvalue < 0.05:
        print("Significant: YES")
    else:
        print("Significant: NO")


if __name__ == "__main__":
    main()