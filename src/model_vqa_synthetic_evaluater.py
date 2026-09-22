import argparse
import json
import re


def normalize_prediction(text):
    text = text.strip().upper()

    # Ideal case
    if len(text) == 1 and text.isalnum():
        return text

    # Common LLaVA responses:
    # "The character is Y."
    # "It is Y."
    # "Y."
    patterns = [
        r"(?:character|letter|number|digit)\s+is\s+([A-Z0-9])",
        r"(?:it|answer)\s+is\s+([A-Z0-9])",
        r"^([A-Z0-9])[\.\,\!\?]?$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return match.group(1)

    return text


def main(args):

    total = 0
    correct = 0

    with open(args.answers_file, "r") as f:

        for line in f:

            sample = json.loads(line)

            pred = normalize_prediction(
                sample["prediction"]
            )

            gt = sample["answer"].strip().upper()

            total += 1
            correct += int(pred == gt)

    accuracy = correct / total if total else 0.0

    print("=" * 50)
    print("Synthetic Patch OCR")
    print("=" * 50)
    print(f"Correct:  {correct}")
    print(f"Total:    {total}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--answers-file",
        required=True,
    )

    args = parser.parse_args()

    main(args)