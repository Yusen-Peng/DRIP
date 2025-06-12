import torch
from torch.utils.data import DataLoader
import math
from datasets import load_dataset
from PIL import Image
import requests
import io
import os
import torch
import logging
from torch.utils.data import IterableDataset

class LAIONStreamingDataset(IterableDataset):
    def __init__(self, tokenizer, image_transform, max_samples=None):
        self.dataset = load_dataset("laion/laion400m", split="train", streaming=True)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_samples = max_samples
        pid = os.getpid()
        os.makedirs("tmp", exist_ok=True)  # ✅ create the directory if it doesn't exist
        logging.basicConfig(
            filename=f"tmp/laion_worker_{pid}.log",
            level=logging.INFO,
            format="%(asctime)s [PID %(process)d] %(message)s",
            force=True  # ✅ resets logger config inside worker process
        )
        self.logger = logging.getLogger(f"worker-{pid}")

    def is_valid(self, example):
        return (
            example.get("NSFW", "") == "UNLIKELY"
            and example.get("similarity", 0.0) > 0.3
            and "caption" in example
            and example.get("url", "").startswith("http")
        )

    def download_image(self, url):
        try:
            r = requests.get(url, timeout=5)
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            return img
        except Exception:
            return None

    def __iter__(self):
        count = 0
        for example in self.dataset:
            if self.max_samples and count >= self.max_samples:
                break
            if not self.is_valid(example):
                continue

            img = self.download_image(example["url"])
            if img is None:
                continue

            try:
                image_tensor = self.image_transform(img)
                text_tensor = (self.tokenizer([str(example["caption"])])[0]).clone().detach()
                count += 1

                yield (image_tensor, text_tensor)
            except Exception as e:
                self.logger.error(f"Error processing example: {example['url']} | {repr(e)}")
                continue

class LAIONStreamingDatasetWrapper:
    def __init__(self, args, preprocess_fn, is_train=True, epoch=0, tokenizer=None):
        # train_num_samples controls how many samples to load from the LAION dataset "on the fly"
        assert is_train, "Only training supported for LAION streaming"
        self.dataset = LAIONStreamingDataset(
            tokenizer=tokenizer,
            image_transform=preprocess_fn,
            max_samples=args.train_num_samples
        )
        self.dataloader: DataLoader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True
        )
        
        self.num_samples = args.train_num_samples
        self.num_batches = math.ceil(self.num_samples / args.batch_size)
        self.dataloader.num_batches = self.num_batches
        self.dataloader.num_samples = self.num_samples
        self.dataloader_type = "iterable"  # this is CRITICAL for iterable datasets

        print("=" * 40)
        print("[INFO] CONGRATULATIONS! You are using the LAION streaming dataset.")
        print(f"we are processing {self.dataloader.num_samples} in total, buddy!")
        print("LET'S GO!")
        print("=" * 40)
    
    def set_epoch(self, epoch):
        self.epoch = epoch
