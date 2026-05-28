import argparse, json, re
from statistics import mean
from tqdm import tqdm

def levenshtein(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = cur
    return prev[-1]

def normalize_answer(s):
    # DocVQA: case-insensitive, but spaces matter.
    # Keep this conservative. Do NOT aggressively remove punctuation/spaces.
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def anls_one(pred, golds, threshold=0.5):
    pred = normalize_answer(pred)
    if len(pred) == 0:
        return 0.0

    best = 0.0
    for g in golds:
        gold = normalize_answer(g)
        if len(gold) == 0:
            continue
        dist = levenshtein(pred, gold)
        sim = 1.0 - dist / max(len(pred), len(gold))
        if sim < threshold:
            sim = 0.0
        best = max(best, sim)
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-file", required=True)
    ap.add_argument("--pred-file", required=True)
    ap.add_argument("--out-file", default=None)
    args = ap.parse_args()

    gt = json.load(open(args.gt_file))
    preds = {}

    with open(args.pred_file) as f:
        for line in f:
            row = json.loads(line)
            preds[str(row["question_id"])] = row["text"]

    scores = []
    missing = 0

    detailed = {}
    for qid, item in tqdm(gt.items()):
        if qid not in preds:
            missing += 1
            pred = ""
        else:
            pred = preds[qid]

        score = anls_one(pred, item["answers"])
        scores.append(score)

        detailed[qid] = {
            "question": item["question"],
            "prediction": pred,
            "answers": item["answers"],
            "anls": score,
        }

    result = {
        "ANLS": mean(scores),
        "num_examples": len(scores),
        "missing_predictions": missing,
    }

    print(json.dumps(result, indent=2))

    if args.out_file:
        with open(args.out_file, "w") as f:
            json.dump({"summary": result, "details": detailed}, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
