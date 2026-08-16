import argparse
import json
import re
import os
import shutil
from tqdm import tqdm

NUMBER_OF_EXAMPLES = 100


class EvalAIAnswerProcessor:
    NUMBER_MAP = {
        "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9", "ten": "10",
    }
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\",
        "_", "-", ">", "<", "@", "`", ",", "?", "!",
    ]

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or re.search(self.COMMA_STRIP, in_text):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        return self.PERIOD_STRIP.sub("", out_text)

    def process_digit_article(self, in_text):
        out_text = []
        for word in in_text.lower().split():
            word = self.NUMBER_MAP.get(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
        return " ".join(out_text)

    def __call__(self, item):
        item = self.word_tokenize(str(item))
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item


def prompt_processor(prompt):
    if prompt.startswith("OCR tokens: "):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif "Reference OCR token: " in prompt and len(prompt.split("\n")) == 3:
        if prompt.startswith("Reference OCR token:"):
            question = prompt.split("\n")[1]
        else:
            question = prompt.split("\n")[0]
    elif len(prompt.split("\n")) == 2:
        question = prompt.split("\n")[0]
    else:
        raise ValueError(f"Unknown prompt format:\n{prompt}")
    return question.lower()


def compute_textvqa_scores(gt_answers, processor):
    answers = [processor(a) for a in gt_answers]
    assert len(answers) == 10

    gt_answers_enum = list(enumerate(answers))
    unique_scores = {}

    for unique_answer in set(answers):
        accs = []
        for gt in gt_answers_enum:
            others = [item for item in gt_answers_enum if item != gt]
            matches = [item for item in others if item[1] == unique_answer]
            accs.append(min(1.0, len(matches) / 3.0))
        unique_scores[unique_answer] = sum(accs) / len(accs)

    return unique_scores


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_annotations(path):
    ann = json.load(open(path))["data"]
    return {
        (a["image_id"], a["question"].lower()): a
        for a in ann
    }


def build_result_map(result_file):
    results = load_jsonl(result_file)
    out = {}

    for r in results:
        image_id = r["question_id"]
        question = prompt_processor(r["prompt"])
        out[(image_id, question)] = r

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", default="/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/TextVQA_0.5.1_val.json")
    parser.add_argument("--new-downsample-file", default="/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/answers/LLaVA_7B_SigLIP_HF_v2_DRIP_4x_temp01_new_downsample_train_full.jsonl")
    parser.add_argument("--old-downsample-file", default="/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/answers/LLaVA_7B_SigLIP_HF_v2_DRIP_4x_train_full.jsonl")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--min-drip-score", type=float, default=1e-9)
    parser.add_argument("--max-fixed-score", type=float, default=0.0)
    parser.add_argument("--image-dir", default="/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/train_images")
    parser.add_argument("--copy-dir", default="/users/PAS2912/yusenpeng/DRIP/src/example_analysis/TextVQA_results/siglip_diagnose_cases")
    args = parser.parse_args()

    processor = EvalAIAnswerProcessor()

    annotations = load_annotations(args.annotation_file)
    new = build_result_map(args.new_downsample_file)
    old = build_result_map(args.old_downsample_file)

    cases = []

    common_keys = sorted(set(new.keys()) & set(old.keys()))

    for key in tqdm(common_keys):
        image_id, question = key
        ann = annotations.get(key)

        if ann is None:
            continue

        gt_answers = ann["answers"]

        answer_scores = compute_textvqa_scores(gt_answers, processor)

        new_raw = new[key]["text"]
        old_raw = old[key]["text"]

        new_norm = processor(new_raw)
        old_norm = processor(old_raw)

        new_score = answer_scores.get(new_norm, 0.0)
        old_score = answer_scores.get(old_norm, 0.0)

        if old_score >= args.min_drip_score and new_score <= args.max_fixed_score:
            cases.append({
                "question_id": image_id,
                "image_id": image_id,
                "question": question,
                "old_prediction": old_raw,
                "old_score": old_score,
                "new_prediction": new_raw,
                "new_score": new_score,
            })

    print("=" * 80)
    print(f"Found {len(cases)} out of {len(common_keys)} cases where OLD downsampling is correct and NEW downsampling fails.")

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(cases, f, indent=2)
        print(f"Saved to {args.output_file}")
    else:
        print(json.dumps(cases[:NUMBER_OF_EXAMPLES], indent=2))
    
    
    os.makedirs(args.copy_dir, exist_ok=True)
    copied = 0
    for case in tqdm(cases[:NUMBER_OF_EXAMPLES], desc="Copying images"):
        image_id = str(case["image_id"])
        src = os.path.join(args.image_dir, f"{image_id}.jpg")
        if not os.path.exists(src):
            print(f"[WARN] Missing image: {src}")
            continue
        dst = os.path.join(args.copy_dir, f"{image_id}.jpg")
        shutil.copy2(src, dst)
        copied += 1
    print(f"Copied {copied} images to {args.copy_dir}")

if __name__ == "__main__":
    main()
