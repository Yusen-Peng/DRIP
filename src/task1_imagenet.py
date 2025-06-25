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
import random
import numpy as np
from tqdm import trange, tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import get_cosine_schedule_with_warmup


# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

def setup_distributed():
    dist.init_process_group(backend="nccl")  # assumes GPU
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

def cleanup_distributed():
    dist.destroy_process_group()

def distributed_accuracy(correct, total, device):
    correct_tensor = torch.tensor(correct, dtype=torch.long, device=device)
    total_tensor = torch.tensor(total, dtype=torch.long, device=device)

    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    return correct_tensor.item(), total_tensor.item()

def naive_weight_transfer(dtp_vit: nn.Module, clip_vit_state_dict):
    dtp_state_dict = dtp_vit.state_dict()
    transferred = 0

    # map positional embedding
    if "pos_embed" in dtp_state_dict and "positional_embedding" in clip_vit_state_dict:
        dtp_state_dict["pos_embed"].copy_(clip_vit_state_dict["positional_embedding"])
        print("✅ Transferred: pos_embed")

    # map patch embedding
    if "patch_embed.proj.weight" in dtp_state_dict and "conv1.weight" in clip_vit_state_dict:
        dtp_state_dict["patch_embed.proj.weight"].copy_(clip_vit_state_dict["conv1.weight"])
        dtp_state_dict["patch_embed.proj.bias"].zero_()
        print("✅ Transferred: patch_embed.proj")

    # map transformer blocks
    for i in range(12):
        if i < len(dtp_vit.pre_blocks):
            prefix = f"pre_blocks.{i}"
        else:
            j = i - len(dtp_vit.pre_blocks)
            prefix = f"shorten_blocks.{j}"

        base = f"transformer.resblocks.{i}"

        pairs = [
            ("ln_1.weight", "norm1.weight"),
            ("ln_1.bias", "norm1.bias"),
            ("attn.in_proj_weight", "attn.in_proj_weight"),
            ("attn.in_proj_bias", "attn.in_proj_bias"),
            ("attn.out_proj.weight", "attn.out_proj.weight"),
            ("attn.out_proj.bias", "attn.out_proj.bias"),
            ("ln_2.weight", "norm2.weight"),
            ("ln_2.bias", "norm2.bias"),
            ("mlp.c_fc.weight", "mlp.0.weight"),
            ("mlp.c_fc.bias", "mlp.0.bias"),
            ("mlp.c_proj.weight", "mlp.3.weight"),
            ("mlp.c_proj.bias", "mlp.3.bias"),
        ]

        for clip_key, dtp_key in pairs:
            full_clip_key = f"{base}.{clip_key}"
            full_dtp_key = f"{prefix}.{dtp_key}"
            if full_clip_key in clip_vit_state_dict and full_dtp_key in dtp_state_dict:
                if dtp_state_dict[full_dtp_key].shape == clip_vit_state_dict[full_clip_key].shape:
                    dtp_state_dict[full_dtp_key].copy_(clip_vit_state_dict[full_clip_key])
                    transferred += 1
                else:
                    print(f"⚠️ Shape mismatch: {full_clip_key} vs {full_dtp_key}")
                    exit(1)
            else:
                print(f"⛔ Missing: {full_clip_key} or {full_dtp_key}")
                exit(1)
        print("✅ Transferred: " + prefix)  

    print(f"🎉 Transferred {transferred} parameter tensors successfully!")
    print(f"DTP-ViT now has {sum(p.numel() for p in dtp_vit.parameters())} parameters.")

def all_weight_transfer(dtp_vit: nn.Module, clip_vit_state_dict):
    dtp_state_dict = dtp_vit.state_dict()
    transferred = 0

    # map positional embedding
    if "pos_embed" in dtp_state_dict and "positional_embedding" in clip_vit_state_dict:
        src = clip_vit_state_dict["positional_embedding"]      # [50, 768]
        dst = dtp_state_dict["pos_embed"]                       # [1, 50, 768]

        if src.shape == dst.shape[1:]:  # [50, 768] == [50, 768]
            dtp_state_dict["pos_embed"].copy_(src.unsqueeze(0))  # Add batch dim
            print("✅ Transferred: pos_embed")
        else:
            print(f"❌ Cannot transfer pos_embed: shape mismatch {src.shape} → {dst.shape}")    



    # map patch embedding
    if "patch_embed.proj.weight" in dtp_state_dict and "conv1.weight" in clip_vit_state_dict:
        dtp_state_dict["patch_embed.proj.weight"].copy_(clip_vit_state_dict["conv1.weight"])
        dtp_state_dict["patch_embed.proj.bias"].zero_()
        print("✅ Transferred: patch_embed.proj")

    # map transformer blocks
    for i in range(12):
        if i < len(dtp_vit.pre_blocks):
            prefix = f"pre_blocks.{i}"
        else:
            j = i - len(dtp_vit.pre_blocks)
            prefix = f"shorten_blocks.{j}"

        base = f"transformer.resblocks.{i}"

        pairs = [
            ("ln_1.weight", "norm1.weight"),
            ("ln_1.bias", "norm1.bias"),
            ("attn.in_proj_weight", "attn.in_proj_weight"),
            ("attn.in_proj_bias", "attn.in_proj_bias"),
            ("attn.out_proj.weight", "attn.out_proj.weight"),
            ("attn.out_proj.bias", "attn.out_proj.bias"),
            ("ln_2.weight", "norm2.weight"),
            ("ln_2.bias", "norm2.bias"),
            ("mlp.c_fc.weight", "mlp.0.weight"),
            ("mlp.c_fc.bias", "mlp.0.bias"),
            ("mlp.c_proj.weight", "mlp.3.weight"),
            ("mlp.c_proj.bias", "mlp.3.bias"),
        ]

        for clip_key, dtp_key in pairs:
            full_clip_key = f"{base}.{clip_key}"
            full_dtp_key = f"{prefix}.{dtp_key}"
            if full_clip_key in clip_vit_state_dict and full_dtp_key in dtp_state_dict:
                if dtp_state_dict[full_dtp_key].shape == clip_vit_state_dict[full_clip_key].shape:
                    dtp_state_dict[full_dtp_key].copy_(clip_vit_state_dict[full_clip_key])
                    transferred += 1
                else:
                    print(f"⚠️ Shape mismatch: {full_clip_key} vs {full_dtp_key}")
                    exit(1)
            else:
                print(f"⛔ Missing: {full_clip_key} or {full_dtp_key}")
                exit(1)
        print("✅ Transferred: " + prefix)  

    print(f"🎉 Transferred {transferred} parameter tensors successfully!")
    print(f"DTP-ViT now has {sum(p.numel() for p in dtp_vit.parameters())} parameters.")

FREEZE_BACKBONE = False

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

def finetuning_ViT():
    BATCH_SIZE = 512
    EPOCHS = 10
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
    model = DDP(model, device_ids=[DEVICE])

    if FREEZE_BACKBONE:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("🔒 Backbone frozen. Only classification head will be trained.")

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

def training_ViT_from_scratch():
    BATCH_SIZE = 512
    EPOCHS = 30
    # tune the learning rate
    #LR = 1e-4
    LR = 6e-4

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device(f"cuda:{local_rank}")
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
    model = DDP(model, device_ids=[DEVICE])

    if FREEZE_BACKBONE:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("🔒 Backbone frozen. Only classification head will be trained.")

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

    # Final evaluation (DDP-aware)
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

    # Aggregate across all GPUs
    correct_tensor = torch.tensor(correct, dtype=torch.long, device=DEVICE)
    total_tensor = torch.tensor(total, dtype=torch.long, device=DEVICE)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    if dist.get_rank() == 0:
        val_acc = correct_tensor.item() / total_tensor.item()
        print("⭐" * 20)
        print(f"Final Validation Accuracy: {val_acc:.4f}")
        print("⭐" * 20)


def training_DTP_ViT_from_scratch():
    BATCH_SIZE = 512
    NUM_CLASSES = 1000
    EPOCHS = 30
    
    # tune the learning rate
    #LR = 1e-4
    LR = 6e-4
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device(f"cuda:{local_rank}")


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
    model = DDP(model, device_ids=[DEVICE])

    if FREEZE_BACKBONE:
        for name, param in model.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False
        print("🔒 Backbone frozen. Only classification head will be trained.")

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
        model.train()
        total_loss = 0
        correct = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                logits, boundary_loss = model(images, return_loss=True)
                ce_loss = criterion(logits, labels)
                loss = ce_loss + boundary_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # ⬅️ Step the LR scheduler every batch

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()

        train_acc = correct / len(train_loader.dataset)
        if dist.get_rank() == 0:
            print(f"✅ Epoch {epoch}: Train Loss {total_loss:.2f}, Train Acc {train_acc:.4f}")


    print("⭐" * 20)
    print("Finetuning complete!")
    print("⭐" * 20)

    # Final evaluation (DDP-aware)
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

    # Aggregate across all GPUs
    correct_tensor = torch.tensor(correct, dtype=torch.long, device=DEVICE)
    total_tensor = torch.tensor(total, dtype=torch.long, device=DEVICE)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    if dist.get_rank() == 0:
        val_acc = correct_tensor.item() / total_tensor.item()
        print("⭐" * 20)
        print(f"Final Validation Accuracy: {val_acc:.4f}")
        print("⭐" * 20)

if __name__ == "__main__":
    setup_distributed()

    #finetuning_ViT()
    training_ViT_from_scratch()
    #training_DTP_ViT_from_scratch()

    cleanup_distributed()
