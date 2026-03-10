import os
import json
import argparse
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from utils.data_utils import process_single_sample, CAT_SHORT2LONG


def load_predictions(path):
    with open(path, "r") as f:
        preds = json.load(f)

    if not isinstance(preds, dict):
        raise ValueError(
            f"Expected prediction file to be a JSON dict of sample_id -> prediction, got {type(preds)}"
        )
    return preds

def normalize_choice(x):
    if x is None:
        return None
    x = str(x).strip().upper()
    if len(x) == 0:
        return None
    return x[0]  # just in case something weird slips through


def infer_subject_name(sample, fallback="unknown"):
    # Try a few common keys just in case the processed sample keeps different names
    for key in ["subdomain", "subject", "category", "topic", "discipline"]:
        if key in sample and sample[key] is not None:
            val = str(sample[key]).strip()
            if val:
                return val
    return fallback


def build_dataset(data_path, split):
    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values():
        sub_dataset = load_dataset(data_path, subject, split=split)
        sub_dataset_list.append(sub_dataset)
    return concatenate_datasets(sub_dataset_list)


def evaluate(preds, dataset):
    rows = []
    missing = 0

    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    total = 0
    correct = 0

    for raw_sample in tqdm(dataset, desc="Evaluating"):
        sample = process_single_sample(raw_sample)

        sample_id = sample["id"]
        gt = normalize_choice(sample.get("answer"))
        subject = infer_subject_name(sample)

        if sample_id not in preds:
            missing += 1
            continue

        pred = normalize_choice(preds[sample_id])
        hit = int(pred == gt)

        total += 1
        correct += hit
        subject_stats[subject]["correct"] += hit
        subject_stats[subject]["total"] += 1

        rows.append(
            {
                "id": sample_id,
                "subject": subject,
                "gt": gt,
                "pred": pred,
                "correct": hit,
            }
        )

    overall_acc = correct / total if total > 0 else 0.0

    per_subject = {}
    for subject, stats in sorted(subject_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        per_subject[subject] = {
            "accuracy": acc,
            "correct": stats["correct"],
            "total": stats["total"],
        }

    summary = {
        "overall_accuracy": overall_acc,
        "num_evaluated": total,
        "num_correct": correct,
        "num_missing_predictions": missing,
        "per_subject": per_subject,
    }

    detail_df = pd.DataFrame(rows)
    return summary, detail_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate HF MMMU predictions")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to saved prediction JSON")
    parser.add_argument("--data_path", type=str, default="MMMU/MMMU", help="HF dataset path")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--summary_out", type=str, default=None, help="Optional JSON summary output")
    parser.add_argument("--detail_out", type=str, default=None, help="Optional CSV detail output")
    args = parser.parse_args()

    print("Loading predictions...")
    preds = load_predictions(args.pred_path)
    print(f"Loaded {len(preds)} predictions")

    print("Loading MMMU dataset...")
    if args.split == "mixed":
        dataset_dev = build_dataset(args.data_path, "dev")
        dataset_val = build_dataset(args.data_path, "validation")
        dataset = concatenate_datasets([dataset_dev, dataset_val])
    else:
        dataset = build_dataset(args.data_path, args.split)
    print(f"Loaded {len(dataset)} total MMMU samples")

    print("Running evaluation...")
    summary, detail_df = evaluate(preds, dataset)

    print("\n" + "=" * 60)
    print("MMMU Evaluation Results")
    print("=" * 60)
    print(
        f"Overall accuracy: {summary['overall_accuracy']:.4f} "
        f"({summary['num_correct']}/{summary['num_evaluated']})"
    )
    print(f"Missing predictions: {summary['num_missing_predictions']}")

if __name__ == "__main__":
    main()