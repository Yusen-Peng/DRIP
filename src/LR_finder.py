import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from open_clip_local import create_model_and_transforms
from open_clip_local.model import DTPViT
from open_clip_local import CLIP
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import os
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import trange, tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import get_cosine_schedule_with_warmup
from torch_lr_finder import LRFinder

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

class VisionClassifier(nn.Module):
    def __init__(self, backbone, num_classes, DTP_ViT=False):
        super().__init__()
        self.DTP_ViT = DTP_ViT
        self.backbone = backbone
        if not DTP_ViT:
            self.fc = nn.Linear(backbone.output_dim, num_classes)

    def forward(self, x):
        if self.DTP_ViT:
            return self.backbone(x)
        else: 
            feats = self.backbone(x)
            return self.fc(feats)


def find_ViT_from_scratch():
    BATCH_SIZE = 512
    NUM_CLASSES = 1000
    EPOCHS = 30
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    backbone, _, preprocess = create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained=None, # TRAINING IT FROM SCRATCH
        DTP_ViT=False
    )

    train_root = "/fs/scratch/PAS2836/yusenpeng_dataset/train"
    val_root   = "/fs/scratch/PAS2836/yusenpeng_dataset/val"

    train_dataset = datasets.ImageFolder(train_root, transform=preprocess)
    val_dataset   = datasets.ImageFolder(val_root, transform=preprocess)

    NUM_CLASSES = len(train_dataset.classes)
    print("⭐" * 20)
    print(f"Number of classes: {NUM_CLASSES}")
    print("⭐" * 20)

    model = VisionClassifier(backbone.visual, NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=8
    )

    lr_finder = LRFinder(model, optimizer, criterion, device=DEVICE)
    lr_finder.range_test(train_loader, end_lr=1e-3, num_iter=100, step_mode="exp")
    lr_finder.plot()
    plt.savefig("lr_finder_plot_ViT.png")


def find_DTP_ViT_from_scratch():
    BATCH_SIZE = 512
    NUM_CLASSES = 1000
    EPOCHS = 30
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compression_rate = 0.1
    model_backbone = DTPViT(
        image_size=224,
        patch_size=32,
        in_chans=3,
        embed_dim=768,
        depth=(2, 10, 0),
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        temp=0.5,
        compression_rate=compression_rate,
        threshold=0.5,
        activation_function="gelu",
        num_classes=NUM_CLASSES,
    )

    # load CLIP ViT-B/32 weights
    clip_model, _, preprocess = create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        DTP_ViT=False # NOTE: set it to False in order to load pretrained ViT-B-32 weights
    )
    clip_state_dict = clip_model.visual.state_dict()
    clip_state_dict = {
        k: v for k, v in clip_state_dict.items()
        if not k.startswith("proj") and not k.startswith("ln_post") and "attn_mask" not in k
    }

    train_root = "/fs/scratch/PAS2836/yusenpeng_dataset/train"
    val_root   = "/fs/scratch/PAS2836/yusenpeng_dataset/val"

    train_dataset = datasets.ImageFolder(train_root, transform=preprocess)
    val_dataset   = datasets.ImageFolder(val_root, transform=preprocess)

    NUM_CLASSES = len(train_dataset.classes)
    print("⭐" * 20)
    print(f"Number of classes: {NUM_CLASSES}")
    print("⭐" * 20)

    model = model_backbone.to(DEVICE)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=8
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    lr_finder = LRFinder(model, optimizer, criterion, device=DEVICE)
    lr_finder.range_test(train_loader, end_lr=1e-3, num_iter=100, step_mode="exp")
    lr_finder.plot()  # log-scale plot of LR vs loss
    plt.savefig("lr_finder_plot_DTP_ViT.png")
    lr_finder.reset()


if __name__ == "__main__":
    #find_ViT_from_scratch()
    find_DTP_ViT_from_scratch()
