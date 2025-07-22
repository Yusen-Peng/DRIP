import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from open_clip_local import create_model_and_transforms
from open_clip_local.model import DTPViT, VisionTransformer
from boundary_vis import load_dtpx_from_clip_checkpoint
from open_clip_local import CLIP
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import os
import random
import numpy as np
from tqdm import trange, tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import get_cosine_schedule_with_warmup


FREEZE_BACKBONE = False


# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

# def setup_distributed():
#     dist.init_process_group(backend="nccl")
#     torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

def setup_distributed():
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_distributed():
    dist.destroy_process_group()

def distributed_accuracy(correct, total, device):
    correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
    total_tensor = torch.tensor(total, dtype=torch.long, device=device)

    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    return correct_tensor.item(), total_tensor.item()

class VisionClassifier(nn.Module):
    def __init__(self, backbone: DTPViT | VisionTransformer, num_classes):
        super().__init__()
        self.backbone = backbone
        if isinstance(backbone, DTPViT):
            self.fc = nn.Linear(backbone.num_classes, num_classes)
        elif isinstance(backbone, VisionTransformer):
            self.fc = nn.Linear(backbone.output_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.fc(feats)

def finetuning_ViT():
    BATCH_SIZE = 512
    EPOCHS = 100
    LR = 1e-4
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device(f"cuda:{local_rank}")


    backbone, _, preprocess = create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
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
    if FREEZE_BACKBONE:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("🔒 Backbone frozen. Only classification head will be trained.")
    model = DDP(model, device_ids=[DEVICE])

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.05 * total_steps)  # 5% warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5  # Default: cosine from max → 0
    )
    scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))

    for epoch in trange(EPOCHS, desc="Training Epochs"):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        correct = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # ⬅️ Step the LR scheduler every batch

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_loader.dataset)
        if dist.get_rank() == 0:
            print(f"✅ Epoch {epoch}: Train Loss {total_loss:.2f}, Train Acc {train_acc:.4f}")


    print("⭐" * 20)
    print("Finetuning complete!")
    print("⭐" * 20)

    # Final evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"[TEST]", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with autocast():
                outputs = model(images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print("⭐" * 20)
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    print("⭐" * 20)

def finetuning_DTP_ViT():
    BATCH_SIZE = 64
    EPOCHS = 100
    effective_batch_size = BATCH_SIZE * dist.get_world_size()
    LR = 1e-3 * (effective_batch_size / 512)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device(f"cuda:{local_rank}")

    compression = "2x"  # "2x", "4x", or "10x"
    patch_size = 16
    epoch_checkpoint = 3  # load checkpoint from epoch
    if compression == "2x":
        compression_rate = 0.5
    elif compression == "4x":
        compression_rate = 0.25
    elif compression == "10x":
        compression_rate = 0.1
    
    _, _, preprocess = create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
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
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True, persistent_workers=True)

    empty_backbone = DTPViT(
            image_size=224,
            patch_size=patch_size,
            embed_dim=768,
            num_heads=12,
            depth=(2, 10, 0),
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.1,
            num_classes=512,
            temp=0.5,
            compression_rate=compression_rate,
            threshold=0.5,
            activation_function='gelu',
            flop_measure=False
        )

    #ckpt_path = f"logs/DTP-ViT-{compression}-{patch_size}/checkpoints/epoch_{epoch_checkpoint}.pt"
    # FIXME: this is for temporay testing, should be removed later
    ckpt_path = f"logs/DRIP-2X-16/checkpoints/epoch_{epoch_checkpoint}.pt"
    backbone = load_dtpx_from_clip_checkpoint(empty_backbone, ckpt_path)

    model = VisionClassifier(backbone, NUM_CLASSES).to(DEVICE)
    if FREEZE_BACKBONE:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("🔒 Backbone frozen. Only classification head will be trained.")
    model = DDP(model, device_ids=[DEVICE])    

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.05 * total_steps)  # 5% warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5
    )
    scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))

    for epoch in trange(EPOCHS, desc="Training Epochs"):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        correct = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # ⬅️ Step the LR scheduler every batch

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_loader.dataset)
        if dist.get_rank() == 0:
            print(f"\n✅ Epoch {epoch}: Train Loss {total_loss:.2f}, Train Acc {train_acc:.4f}", flush=True)

    # Final evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"[TEST]", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with autocast():
                outputs = model(images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print("⭐" * 20)
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    print("⭐" * 20)


if __name__ == "__main__":
    setup_distributed()
    finetuning_DTP_ViT()
    dist.barrier()
    cleanup_distributed()
