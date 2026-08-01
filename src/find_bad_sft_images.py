import os
import json
from PIL import Image

json_path = "/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_finetuning/cleaned.json"
image_root = "/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_finetuning"

with open(json_path, "r") as f:
    data = json.load(f)

bad = []

for i, sample in enumerate(data):
    if "image" not in sample:
        continue
    img_path = os.path.join(image_root, sample["image"])
    try:
        with Image.open(img_path) as img:
            img.verify()   # verify integrity
        with Image.open(img_path) as img:
            img.convert("RGB")
    except Exception as e:
        bad.append((i, sample["image"], str(e)))
        print(f"BAD: idx={i}, file={sample['image']}, err={e}")

print(f"\nTotal bad images: {len(bad)}")