import json
import argparse

official_path = "/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/OCRBench_v2.json"



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Path to the prediction JSON file.")
    parser.add_argument("--out", required=True, help="Path to the output merged JSON file.")
    args = parser.parse_args()


    pred = json.load(open(args.pred))
    official = json.load(open(official_path))
    official_map = {
        (x["id"], x["dataset_name"], x["type"]): x
        for x in official
    }

    merged = []

    for p in pred:
        key = (p["id"], p["dataset_name"], p["type"])
        base = official_map[key].copy()
        base["predict"] = p["predict"]
        merged.append(base)

    json.dump(merged, open(args.out, "w"), indent=4, ensure_ascii=False)

    print("saved", args.out)