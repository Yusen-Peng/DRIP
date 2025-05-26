from open_clip_train_local.main import main as train_function

def train_runner(
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
        train_num_samples: int | None = 10000000,  # only used if webdataset is True
        imagenet_val_path: str = "dataset/ImageNet_val"
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
            "--model", model
        ]

    train_function(args_list)

def main():
    # dataset parameters
    dataset_name = "LAION"  # Options: "COCO", "LAION"

    if dataset_name == "COCO":
        # COCO dataset
        use_webdataset = False
        train_data_path = "dataset/coco_train_subset.csv"
        val_data_path = "dataset/coco_val_subset.csv"
        train_num_samples = None

    elif dataset_name == "LAION":
        # LAION dataset
        use_webdataset = True
        train_data_path = "::".join([f"dataset/laion_shards/laion-{i:06d}.tar" for i in range(8)])

        train_data_path = ""
        for i in range(7):
            train_data_path += f"dataset/laion_shards/laion-00000{i}.tar::"
        train_data_path += f"dataset/laion_shards/laion-000007.tar"


        val_data_path = ""
        for i in range(20):
            val_data_path += f"dataset/laion_shards/laion-00000{i}.tar::"
        val_data_path += f"dataset/laion_shards/laion-000020.tar"

        train_num_samples = 80_000 # only used if webdataset is True


    print(f"training datapath: {train_data_path}")
    print(f"validation datapath: {val_data_path}")

    
    # training parameters
    warmup = 50
    batch_size = 32
    lr = 1e-4
    wd = 0.1
    epochs = 30 # 1, 10, 30
    workers = 1
    model = "RN50"

    # train CLIP
    train_runner(
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
    print("Training completed.")


if __name__ == "__main__":
    main() 