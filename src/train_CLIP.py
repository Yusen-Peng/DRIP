import torch
import torch.distributed as dist
import os
import glob
from open_clip_train_local.main import main as train_function

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
        imagenet_val_path: str = "dataset/ImageNet_val",
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
            #"--dataset-type", "laion-stream",
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
        ]

    if DTP:
        args_list.append("--DTP")
    train_function(args_list)

def main():
    # dataset parameters
    # now we are running a 1M subset of the LAION-400M dataset
    dataset_name = "LAION"  # "COCO" or "LAION"

    # experiment with batch size
    use_DTP = True # DTP (Dynamic Token Pruning) is not used by default

    # batch size:
    # 1024 for ViT-B-32
    batch_size = 512
    lr = 1e-4
    wd = 0.1
    epochs = 2
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
        # LAION dataset (for now, it's 1M subset of LAION-400M)
        use_webdataset = True

        #train_data_path = "::".join(sorted(glob.glob("dataset/laion_shards/laion-*.tar")))
        #val_data_path = "::".join(sorted(glob.glob("dataset/laion_val/laion-*.tar")))
        
        # FIXME: stream-mode: "load on the fly" - bad idea
        # train_data_path = "laion-stream"  # just a keyword
        # val_data_path = None  # no val needed for now

        # FIXME: 1M subset of LAION-400M for now 
        train_num_samples = 1_000_000

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