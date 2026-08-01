"""
python src/DocVQA_eval/prepare_docvqa_llava.py \
  --split validation \
  --out-dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa
"""


import argparse, json, os
from datasets import load_dataset
from tqdm import tqdm

def get_field(x, names):
    for n in names:
        if n in x:
            return x[n]
    raise KeyError(f"Missing any of fields: {names}. Available: {list(x.keys())}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lmms-lab/DocVQA")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompt-style", default="short", choices=["raw", "short"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    image_dir = os.path.join(args.out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    qfile = os.path.join(args.out_dir, f"docvqa_{args.split}_llava.jsonl")
    gtfile = os.path.join(args.out_dir, f"docvqa_{args.split}_gt.json")

    ds = load_dataset(args.dataset, "DocVQA", split=args.split)

    gt = {}
    with open(qfile, "w") as f:
        for i, ex in enumerate(tqdm(ds)):
            qid = get_field(ex, ["questionId", "question_id", "id"])
            question = get_field(ex, ["question", "query"])
            answers = get_field(ex, ["answers", "answer"])

            if isinstance(answers, str):
                answers = [answers]

            image = get_field(ex, ["image"])
            image_name = f"{qid}.png"
            image.save(os.path.join(image_dir, image_name))

            if args.prompt_style == "short":
                text = question.strip() + "\nAnswer the question using a short phrase."
            else:
                text = question.strip()

            row = {
                "question_id": str(qid),
                "image": image_name,
                "text": text,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            gt[str(qid)] = {
                "question": question,
                "answers": answers,
                "image": image_name,
            }

    with open(gtfile, "w") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)

    print("Wrote:", qfile)
    print("Wrote:", gtfile)
    print("Images:", image_dir)


if __name__ == "__main__":
    main()
