#!/usr/bin/env python
# Efficient LAION-400M downloader: 40M samples, production-grade
# Usage: run on SLURM or manually with mp.spawn/run_parallel_download()

import os
import io
import random
import requests
import multiprocessing as mp
from PIL import Image
from datasets import load_dataset
import webdataset as wds

def is_valid(example):
    return (
        example.get("NSFW", "") == "UNLIKELY"
        and example.get("similarity", 0.0) > 0.3
        and "caption" in example
        and example.get("url", "").startswith("http")
    )

def download_and_write(proc_id, total_per_proc, samples_per_shard, output_dir):
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/val", exist_ok=True)

    # Load and shard dataset manually to avoid overlap
    dataset = load_dataset("laion/laion400m", split="train", streaming=True)

    OVERSAMPLE = 30  # adjust based on observed yield rate (e.g., 3–5%)
    skip_amt = proc_id * total_per_proc * OVERSAMPLE
    take_amt = total_per_proc * OVERSAMPLE
    dataset = dataset.skip(skip_amt).take(take_amt)

    train_writer = wds.ShardWriter(f"{output_dir}/train/laion-{proc_id:03d}-%06d.tar", maxcount=samples_per_shard)
    val_writer = wds.ShardWriter(f"{output_dir}/val/laion-{proc_id:03d}-%06d.tar", maxcount=samples_per_shard)

    count = 0
    
    max_attempts = total_per_proc * 20  # assume at worst 5% yield

    for i, example in enumerate(dataset):
        if i >= max_attempts:
            print(f"[Proc {proc_id}] ⚠️ Max attempts reached, only {count} samples collected.", flush=True)
            break
        if not is_valid(example):
            continue
        try:
            r = requests.get(example["url"], timeout=5)
            img = Image.open(io.BytesIO(r.content))
            img.verify()  # Check if corrupted
            img = Image.open(io.BytesIO(r.content)).convert("RGB")

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")

            key = f"{proc_id:03d}_{count:08d}"
            sample = {
                "__key__": key,
                "jpg": buffer.getvalue(),
                "txt": example["caption"].encode("utf-8")
            }

            if random.random() < 0.98:
                train_writer.write(sample)
            else:
                val_writer.write(sample)

            count += 1
            if count % 100 == 0:
                print(f"[Proc {proc_id}] ✅ {count} samples written...", flush=True)
            if count >= total_per_proc:
                break

        except Exception as e:
            continue

    train_writer.close()
    val_writer.close()
    print(f"[Proc {proc_id}] 🚀 Done: {count} samples total.", flush=True)

def run_parallel_download(total_samples=40_000_000, num_processes=64, samples_per_shard=10_000, output_dir="dataset/laion_40M"):
    os.makedirs(output_dir, exist_ok=True)
    samples_per_proc = total_samples // num_processes
    tasks = [(i, samples_per_proc, samples_per_shard, output_dir) for i in range(num_processes)]

    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(download_and_write, tasks)

if __name__ == "__main__":
    total_samples = 40_000     # e.g., 40M in production, 40_000_000
    num_processes = 64
    samples_per_shard = 10_000
    output_dir = "/fs/scratch/PAS2836/yusenpeng"

    # Create output dir and dummy file to ensure existence
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "dummy.txt"), "w") as f:
        f.write("This is a dummy file to ensure the output directory exists.")

    run_parallel_download(
        total_samples=total_samples,
        num_processes=num_processes,
        samples_per_shard=samples_per_shard,
        output_dir=output_dir
    )

    os.remove(os.path.join(output_dir, "dummy.txt"))