import os
import io
import random
import requests
from PIL import Image
from datasets import load_dataset
import webdataset as wds

if __name__ == "__main__":
    # Output directories
    TRAIN_DIR = "dataset/laion_shards"
    VAL_DIR = "dataset/laion_val"
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    # Sharding config
    TOTAL_SAMPLES = 1_000_000 # 1M samples

    # FIXME: debugging - let's do 100k first
    TOTAL_SAMPLES = 100_000
    print("=" * 50)
    print(f"we are loading {TOTAL_SAMPLES} samples...")
    print("=" * 50)


    SAMPLES_PER_SHARD = 10_000
    VAL_RATIO = 0.2  # 80% train, 20% val: 80k train, 20k val
    train_writer = wds.ShardWriter(os.path.join(TRAIN_DIR, "laion-%06d.tar"), maxcount=SAMPLES_PER_SHARD)
    val_writer = wds.ShardWriter(os.path.join(VAL_DIR, "laion-val-%06d.tar"), maxcount=1000)

    print("📡 Loading LAION-400M metadata (streaming)...")
    dataset = load_dataset("laion/laion400m", split="train", streaming=True)

    def is_valid(example):
        return (
            example.get("NSFW", "") == "UNLIKELY" and
            example.get("similarity", 0.0) > 0.3 and
            "caption" in example and
            example.get("url", "").startswith("http")
        )

    def download_image(url):
        try:
            r = requests.get(url, timeout=5)
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            return img
        except Exception:
            return None

    print(f"📦 Writing WebDataset train/val shards...")
    sample_count = 0
    for idx, example in enumerate(dataset):
        if not is_valid(example):
            continue

        img = download_image(example["url"])
        if img is None:
            continue

        buffer = io.BytesIO()
        try:
            img.save(buffer, format="JPEG")
        except Exception:
            continue

        key = f"{sample_count:08d}"
        sample = {
            "__key__": key,
            "jpg": buffer.getvalue(),
            "txt": example["caption"].encode("utf-8"),
        }

        if random.random() < VAL_RATIO:
            val_writer.write(sample)
        else:
            train_writer.write(sample)

        sample_count += 1
        if sample_count % 1000 == 0:
            print(f"✅ Processed {sample_count} samples...")

        if sample_count >= TOTAL_SAMPLES:
            break

    train_writer.close()
    val_writer.close()
    print(f"\n🎉 Finished: {sample_count} samples written (train + val).")
