from open_clip_train_local.main import main as eval_function


"""
python train.py --model ViT-B-32 --pretrained laion2b_s34b_b79k --val-data src/dataset/ImageNet_val --imagenet-val
"""

def eval_runner(
        batch_size: int = 128,
        model: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        imagenet_val_path: str = "dataset/ImageNet_val"
    ):
    
    args_list = [
        "--batch-size", str(batch_size),
        "--model", model,
        "--pretrained", pretrained,
        "--imagenet-val", imagenet_val_path
    ]

    eval_function(args_list)

def main():
    # training parameters
    batch_size = 32
    
    #model = "ViT-B-32"
    #model = "ViT-B-16"
    # model = "ViT-L-14"
    model = "RN50x16"

    #pretrained = "laion2b_s34b_b79k" # for ViT-B-32
    #pretrained = "laion2b_s34b_b88k"  # for ViT-B-16
    #pretrained = "laion2b_s32b_b82k" # for ViT-L-14
    pretrained = "openai" # for RN50x16
    
    imagenet_val_path = "dataset/ImageNet_val"

    # train CLIP
    eval_runner(
        batch_size=batch_size,
        model=model,
        pretrained=pretrained,
        imagenet_val_path=imagenet_val_path
    )
    print("Evaluation completed.")


if __name__ == "__main__":
    main() 