import argparse
import ast
import json
import re
from typing import Any, List, Optional

import numpy as np
from anls import anls_score
from statsmodels.stats.contingency_tables import mcnemar


def load_predictions(file_path):
    if file_path.endswith(".jsonl"):
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def fix_list_format(item: str) -> Any:
    if not isinstance(item, str):
        return item

    match = re.match(r"^\[(.*)\]$", item.strip())
    if not match:
        return item

    content = match.group(1)
    corrected = re.sub(r"(?<!['\w])(\w[^,]*?)(?!['\w])", r"'\1'", content)

    try:
        return ast.literal_eval(f"[{corrected}]")
    except (SyntaxError, ValueError):
        return item


def parse_to_list(text: str) -> Optional[List[str]]:
    if not isinstance(text, str):
        return None

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None

    if isinstance(parsed, list):
        return [str(x).strip(" '") for x in parsed]

    return None


def to_float(text: str) -> Optional[float]:
    try:
        return float(str(text).strip().strip("%"))
    except ValueError:
        return None


def evaluate_single_answer(target, prediction, max_relative_change=0.05):
    t = str(target).strip().strip("%").strip()
    p = str(prediction).strip().strip("%").strip()

    t_f = to_float(t)
    p_f = to_float(p)

    if t_f is not None and p_f is not None:
        if t_f == 0.0:
            return 1.0 if p_f == 0.0 else 0.0

        change = abs(p_f - t_f) / abs(t_f)
        return 1.0 if change <= max_relative_change else 0.0

    return anls_score(
        prediction=p.lower(),
        gold_labels=[t.lower()],
        threshold=0.5,
    )


def relaxed_correctness_chartqapro(
    target,
    prediction,
    max_relative_change=0.05,
    year_flags=None,
    always_use_exact_match=False,
):
    fixed_t = fix_list_format(target)
    t_list = parse_to_list(str(fixed_t)) or [str(target)]
    p_list = parse_to_list(str(prediction)) or [str(prediction)]

    n = len(t_list)

    if year_flags is not None and len(year_flags) < n:
        year_flags = year_flags * n

    scores = []

    for idx in range(max(len(t_list), len(p_list))):
        if idx >= len(t_list) or idx >= len(p_list):
            scores.append(0.0)
            continue

        t_item = t_list[idx]
        p_item = p_list[idx]

        if year_flags is None:
            flag = "NO"
        else:
            flag = year_flags[idx]

        flag_cond = True if str(flag).upper() == "YES" else False

        if flag_cond or always_use_exact_match:
            scores.append(
                1.0
                if str(t_item).strip().lower() == str(p_item).strip().lower()
                else 0.0
            )
        else:
            scores.append(
                evaluate_single_answer(
                    t_item,
                    p_item,
                    max_relative_change,
                )
            )

    return sum(scores) / len(scores) if scores else 0.0


def merge_llava_jsonl_with_gt(pred_file, gt_file):
    predictions = load_predictions(pred_file)

    if not pred_file.endswith(".jsonl"):
        return predictions

    gt = load_predictions(gt_file)

    pred_map = {
        str(x["question_id"]): x["text"]
        for x in predictions
    }

    merged = []

    for item in gt:
        qid = str(item["question_id"])
        cur = dict(item)
        cur["prediction"] = pred_map.get(qid, "")
        merged.append(cur)

    return merged


def load_chartqapro_scores(pred_file, gt_file):
    predictions = merge_llava_jsonl_with_gt(pred_file, gt_file)

    scores = {}
    correctness = {}

    for idx, item in enumerate(predictions):
        qid = str(item.get("question_id", idx))

        gt = item["Answer"][-1].strip(".").strip("\n")
        pred = str(item["prediction"]).strip(".").strip("\n")
        split = item["Question Type"]
        year_flags = item["Year"]

        if split == "Conversational":
            year_flags = year_flags[-1:]

        always_use_exact_match = split in [
            "Fact Checking",
            "Multi Choice",
        ]

        score = relaxed_correctness_chartqapro(
            gt,
            pred,
            year_flags=year_flags,
            always_use_exact_match=always_use_exact_match,
        )

        scores[qid] = score
        correctness[qid] = int(score > 0.0)

    return correctness, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", required=True)
    parser.add_argument("--baseline-file", required=True)
    parser.add_argument("--method-file", required=True)
    parser.add_argument("--exact", action="store_true")
    args = parser.parse_args()

    baseline, baseline_scores = load_chartqapro_scores(
        args.baseline_file,
        args.gt_file,
    )

    method, method_scores = load_chartqapro_scores(
        args.method_file,
        args.gt_file,
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

    baseline_score = np.mean([baseline_scores[qid] for qid in common_ids]) * 100
    method_score = np.mean([method_scores[qid] for qid in common_ids]) * 100

    baseline_bin = (a + b) / len(common_ids) * 100
    method_bin = (a + c) / len(common_ids) * 100

    print("=" * 70)
    print("ChartQAPro McNemar Test")
    print("=" * 70)
    print(f"N: {len(common_ids)}")
    print("Binarization: relaxed score > 0.0")
    print()

    print("Contingency table")
    print()
    print("                         Method correct    Method wrong")
    print(f"Baseline correct      {a:>8}            {b:>8}")
    print(f"Baseline wrong        {c:>8}            {d:>8}")
    print()

    print(f"Baseline score: {baseline_score:.2f}")
    print(f"Method score:   {method_score:.2f}")
    print(f"Score delta:    {method_score - baseline_score:+.2f}")
    print()

    print(f"Baseline binary accuracy: {baseline_bin:.2f}")
    print(f"Method binary accuracy:   {method_bin:.2f}")
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
