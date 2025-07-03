import torch
import torch.distributed as dist
import os
import glob
import json
from open_clip_train_local.main import main as train_function

def infer_successful_samples_from_stats(stats_glob: str) -> int:
    """
    Sum up 'successes' from all _stats.json files.
    """
    total_successes = 0
    for stat_file in glob.glob(stats_glob):
        with open(stat_file, "r") as f:
            stats = json.load(f)
            total_successes += stats.get("successes", 0)
    return total_successes

def train_runner(
        DTP: bool,
        use_webdataset: bool = False,
        train_data_path: str = "dataset/coco_train_subset.csv",
        val_data_path: str = "dataset/coco_val_subset.csv",
        warmup: int = 1000,
        batch_size: int = 128,
        lr: float = 1e-3,
        wd: float = 0.1,
        epochs: int = 4,
        workers: int = 8,
        model: str = "RN50",
        train_num_samples: int | None = 1_000_000,  # only used if webdataset is True
        imagenet_val_path: str = "/fs/scratch/PAS2836/yusenpeng_dataset/val",
    ):

    args_list = []

    if use_webdataset == True:

        args_list = [
            "--train-data", train_data_path,
            "--dataset-type", "webdataset",
            "--train-num-samples", str(train_num_samples),
            "--imagenet-val", imagenet_val_path,
            "--save-frequency", "1",
            "--zeroshot-frequency", "1",
            "--report-to", "tensorboard",
            "--batch-size", str(batch_size),
            "--warmup", str(warmup),
            "--lr", str(lr),
            "--wd", str(wd),
            "--epochs", str(epochs),
            "--workers", str(workers),
            "--model", model,
            "--precision", "amp"
        ]

    else:
        args_list = [
            "--save-frequency", "1",
            "--zeroshot-frequency", "1",
            "--report-to", "tensorboard",
            "--train-data", train_data_path,
            "--val-data", val_data_path,
            "--csv-img-key", "filepath",
            "--csv-caption-key", "caption",
            "--imagenet-val", imagenet_val_path,
            "--warmup", str(warmup),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
            "--wd", str(wd),
            "--epochs", str(epochs),
            "--workers", str(workers),
            "--model", model,
            "--precision", "amp"
        ]

    if DTP:
        args_list.append("--DTP")
    train_function(args_list)

def main():
    # dataset parameters - "COCO" or "LAION" or "CC12"
    dataset_name = "LAION"

    use_DTP = True  # DTP (Dynamic Token Pruning) is not used by default

    # experiment with batch size
    # batch size:
    # 1024 for ViT-B-32
    batch_size = 512
    lr = 1e-4
    wd = 0.1
    epochs = 10
    workers = 8       # CPU utilization
    model = "ViT-B-32"
    warmup = 50

    if dataset_name == "COCO":
        # COCO dataset
        use_webdataset = False
        train_data_path = "dataset/coco_train_subset.csv"
        val_data_path = "dataset/coco_val_subset.csv"
        train_num_samples = None

    elif dataset_name == "LAION":
        PATH = "/fs/scratch/PAS2836/laion2b-data/"
        use_webdataset = True
        train_data_path = "::".join(sorted(glob.glob(f"{PATH}*.tar")))
        val_data_path = None  # no val needed for now
 
        train_num_samples = infer_successful_samples_from_stats(
            f"{PATH}*_stats.json"
        )
        print("🌝"* 30)
        print(f"Total successful samples in LAION dataset: {train_num_samples}")
        print("🌝"* 30)

    elif dataset_name == "CC12":
        # CC12M dataset
        use_webdataset = True
        train_data_path = "::".join(sorted(glob.glob("/fs/scratch/PAS2836/yusenpeng_dataset/cc12m/*.tar")))
        val_data_path = None

        train_num_samples = infer_successful_samples_from_stats(
            "/fs/scratch/PAS2836/yusenpeng_dataset/cc12m/*_stats.json"
        )
        print("🌝"* 30)
        print(f"Total successful samples in CC12 dataset: {train_num_samples}")
        print("🌝"* 30)

    # train CLIP
    train_runner(
        DTP=use_DTP,
        use_webdataset=use_webdataset,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        warmup=warmup,
        batch_size=batch_size,
        lr=lr,
        wd=wd,
        epochs=epochs,
        workers=workers,
        model=model,
        train_num_samples=train_num_samples,
    )

    if dist.is_initialized():
        dist.destroy_process_group()
    print("Training completed.")


if __name__ == "__main__":
    main() 