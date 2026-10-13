import json
import re
import argparse


def normalize_answer(answer):
    answer = str(answer).lower().strip()

    # Normalize common imaging terminology
    answer = re.sub(r"\bx[\s-]?ray\b", "xray", answer)

    # Normalize number words
    numbers = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }

    for word, number in numbers.items():
        answer = re.sub(rf"\b{word}\b", number, answer)

    # Remove articles
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)

    # Remove punctuation
    answer = re.sub(r"[^\w\s]", "", answer)

    # Normalize whitespace
    answer = " ".join(answer.split())

    return answer


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def evaluate_slake(ground_truth, predictions):

    prediction_map = {
        item["question_id"]: item
        for item in predictions
    }

    results = {
        "OPEN": {"correct": 0, "total": 0},
        "CLOSED": {"correct": 0, "total": 0},
    }

    for question_id, gt in enumerate(ground_truth):

        # English only
        if gt["q_lang"] != "en":
            continue

        answer_type = gt["answer_type"].upper()

        if answer_type not in results:
            continue

        if question_id not in prediction_map:
            print(f"Warning: missing prediction for question_id={question_id}")
            continue

        prediction = normalize_answer(
            prediction_map[question_id]["text"]
        )

        reference = normalize_answer(
            gt["answer"]
        )

        correct = int(prediction == reference)

        results[answer_type]["correct"] += correct
        results[answer_type]["total"] += 1

    print("\n===== SLAKE English Evaluation =====")

    for answer_type in ["OPEN", "CLOSED"]:
        correct = results[answer_type]["correct"]
        total = results[answer_type]["total"]

        accuracy = (
            100 * correct / total
            if total > 0 else 0
        )

        print(
            f"{answer_type:6s}: "
            f"{accuracy:.2f}% "
            f"({correct}/{total})"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ground-truth",
        required=True
    )

    parser.add_argument(
        "--predictions",
        required=True
    )

    args = parser.parse_args()

    ground_truth = load_json(args.ground_truth)
    predictions = load_jsonl(args.predictions)

    evaluate_slake(
        ground_truth,
        predictions
    )
