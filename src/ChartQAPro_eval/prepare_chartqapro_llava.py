import os
import json
import argparse
from PIL import Image
import io
from datasets import load_dataset
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ahmed-masry/ChartQAPro")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    image_dir = os.path.join(args.out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    llava_jsonl = os.path.join(
        args.out_dir,
        f"chartqapro_{args.split}_llava.jsonl"
    )

    gt_json = os.path.join(
        args.out_dir,
        f"chartqapro_{args.split}_gt.json"
    )

    ds = load_dataset(args.dataset, split=args.split)

    gt_entries = []

    with open(llava_jsonl, "w") as f:
        for idx, sample in enumerate(tqdm(ds)):

            # image = sample["image"]
            # image_name = f"{idx}.png"
            # image.save(os.path.join(image_dir, image_name))
            image_bytes = sample["image"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_name = f"{idx}.png"
            image.save(os.path.join(image_dir, image_name))

            question = sample["Question"]

            qid = str(idx)

            row = {
                "question_id": qid,
                "image": image_name,
                "text": question
            }

            f.write(json.dumps(row) + "\n")

            gt_entries.append({
                "question_id": qid,
                "Question": sample["Question"],
                "Answer": sample["Answer"],
                "Question Type": sample["Question Type"],
                "Year": sample["Year"]
            })

    with open(gt_json, "w") as f:
        json.dump(gt_entries, f, indent=2)

    print("Saved:")
    print(llava_jsonl)
    print(gt_json)

if __name__ == "__main__":
    main()
