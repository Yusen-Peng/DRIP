import argparse
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
from collections import OrderedDict
import os
import re
import random
import numpy as np
from tqdm import trange, tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import get_cosine_schedule_with_warmup
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image
import random

# Allow argprse.Namespace for safe weights-only unpickling in PyTorch 2.6+
torch.serialization.add_safe_globals([argparse.Namespace])

def _consume_prefix(sd, prefix):
    if not any(k.startswith(prefix) for k in sd):  # nothing to do
        return sd
    out = OrderedDict()
    for k, v in sd.items():
        out[k[len(prefix):] if k.startswith(prefix) else k] = v
    return out

def _drop_head(sd):
    drop = ("fc.", "head.", "classifier.")
    return {k: v for k, v in sd.items() if not any(k.startswith(p) for p in drop)}

def load_backbone_from_imagenet_checkpoint(model, ckpt_path, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    sd = ckpt.get("model", ckpt)
    for p in ("module.", "backbone."):
        sd = _consume_prefix(sd, p)
    sd = _drop_head(sd)
    # match dtype
    dtype = next(model.parameters()).dtype
    sd = {k: (v.to(dtype) if torch.is_tensor(v) else v) for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=False)
    print(f"[imagenet→DTPViT] missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}")
    if msg.missing_keys:    print("  missing (first 15):", msg.missing_keys[:15])
    if msg.unexpected_keys: print("  unexpected (first 15):", msg.unexpected_keys[:15])
    return model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_dtpx_from_clip_checkpoint(model: nn.Module, ckpt_path: str) -> DTPViT:
    """
    Loads weights into a DTPViT model from a CLIP-style checkpoint.

    Args:
        model (DTPViT): An uninitialized DTPViT model with the correct config.
        ckpt_path (str): Path to a checkpoint with 'module.visual.' prefix in keys.

    Returns:
        model (DTPViT): The same model with loaded weights.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    raw_state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    dtpvit_state_dict = {
        k.replace("module.visual.", ""): v
        for k, v in raw_state_dict.items()
        if k.startswith("module.visual.")
    }

    model.load_state_dict(dtpvit_state_dict, strict=False)
    return model.to("cuda")




def load_dtpx_from_clip_checkpoint(model: nn.Module, ckpt_path: str) -> DTPViT:
    """
    Loads weights into a DTPViT model from a CLIP-style checkpoint.

    Args:
        model (DTPViT): An uninitialized DTPViT model with the correct config.
        ckpt_path (str): Path to a checkpoint with 'module.visual.' prefix in keys.

    Returns:
        model (DTPViT): The same model with loaded weights.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    raw_state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    dtpvit_state_dict = {
        k.replace("module.visual.", ""): v
        for k, v in raw_state_dict.items()
        if k.startswith("module.visual.")
    }

    model.load_state_dict(dtpvit_state_dict, strict=False)
    return model.to("cuda")


def load_dtpx_from_clip_checkpoint_float(model: nn.Module, ckpt_path: str) -> DTPViT:
    """
    Loads weights into a DTPViT model from a CLIP-style checkpoint.

    Args:
        model (DTPViT): An uninitialized DTPViT model with the correct config.
        ckpt_path (str): Path to a checkpoint with 'module.visual.' prefix in keys.

    Returns:
        model (DTPViT): The same model with loaded weights.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    raw_state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    dtpvit_state_dict = {
        k.replace("module.visual.", ""): v.float() 
        for k, v in raw_state_dict.items()
        if k.startswith("module.visual.")
    }

    model.load_state_dict(dtpvit_state_dict, strict=False)
    return model


@torch.no_grad()
def visualize_boundaries_enhanced(model: DTPViT, image_tensor: torch.Tensor, save_path=None):
    import torch
    import torchvision.transforms.functional as TF
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- input ----
    x_img = image_tensor.unsqueeze(0).to(device)  # [B=1, 3, H, W]

    # ---- patch embed: Conv2d -> [B, D, Hg, Wg] ----
    feat = model.patch_embed(x_img)
    B, D, Hg, Wg = feat.shape
    grid_h, grid_w = Hg, Wg                     # grid size for masks

    # tokens [B, L, D] with L = Hg*Wg
    x = feat.flatten(2).transpose(1, 2).contiguous()   # [B, L, D]

    # optional dropout / pos-drop
    drop = getattr(model, "dropout", None) or getattr(model, "pos_drop", None)
    if drop is not None:
        x = drop(x)

    # ---- positional encoding for pre-blocks ----
    # your DTPViT uses a callable pos_emb(seq) that returns [L, 1, D]
    # If your seq length is exactly L (no CLS), make 'pos_seq' of length L:
    L = x.size(1)
    pos_seq = torch.arange(L-1, -1, -1.0, device=x.device, dtype=x.dtype)  # mirrors your code
    r = model.pos_emb(pos_seq)   # [L, 1, D]

    # ---- run pre_blocks (they expect seq-first [L, B, D]) ----
    x = x.transpose(0, 1)  # [L, B, D]
    for block in model.pre_blocks:
        x = block(x, r, model.r_w_bias, model.r_r_bias)  # still [L, B, D]

    # ---- boundary predictor ----
    soft_boundaries, hard_boundaries = model.boundary_predictor(x)  # could be [L,B] or [B,L] depending on impl

    # normalize to [B, L]
    if soft_boundaries.dim() == 2 and soft_boundaries.shape[0] == L:   # [L,B]
        soft_boundaries = soft_boundaries.transpose(0, 1).contiguous() # [B,L]
        hard_boundaries = hard_boundaries.transpose(0, 1).contiguous() # [B,L]

    # No CLS token in this path → use all L tokens
    soft_mask = soft_boundaries[0].detach().cpu().view(grid_h, grid_w).numpy()
    hard_mask = hard_boundaries[0].detach().cpu().view(grid_h, grid_w).numpy()

    # ---- recover original image ----
    orig_img = TF.normalize(image_tensor.clone(),
                            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                            std=[1/0.229, 1/0.224, 1/0.225])
    orig_img = TF.to_pil_image(orig_img.cpu().clamp(0, 1)).convert("RGB")
    orig_np = np.array(orig_img).astype(np.uint8)

    image_size = getattr(model, "image_size", orig_np.shape[0])
    patch_h = image_size // grid_h
    patch_w = image_size // grid_w

    # ---- overlays ----
    import cv2
    cmap = plt.get_cmap("hot")
    heatmap_soft = cv2.resize(soft_mask, (image_size, image_size))
    heatmap_colored = (cmap(heatmap_soft)[..., :3] * 255).astype(np.uint8)
    soft_img = Image.fromarray(heatmap_colored)

    red_overlay_np = orig_np.copy()
    for i in range(grid_h):
        for j in range(grid_w):
            if hard_mask[i, j] > 0.5:
                y0, y1 = i*patch_h, (i+1)*patch_h
                x0, x1 = j*patch_w, (j+1)*patch_w
                patch = red_overlay_np[y0:y1, x0:x1]
                red = np.zeros_like(patch); red[..., 0] = 255
                red_overlay_np[y0:y1, x0:x1] = (0.6*patch + 0.4*red).astype(np.uint8)
    hard_overlay_img = Image.fromarray(red_overlay_np)

    # ---- 1D raster strip ----
    patch_list = []
    for i in range(grid_h):
        for j in range(grid_w):
            y0, y1 = i*patch_h, (i+1)*patch_h
            x0, x1 = j*patch_w, (j+1)*patch_w
            patch = orig_np[y0:y1, x0:x1].copy()
            if hard_mask[i, j] > 0.5:
                red = np.zeros_like(patch); red[..., 0] = 255
                patch = (0.6*patch + 0.4*red).astype(np.uint8)
            patch_list.append(Image.fromarray(patch))

    max_patches = len(patch_list)
    patch_strip = Image.new("RGB", (patch_w * max_patches, patch_h))
    for idx in range(max_patches):
        patch_strip.paste(patch_list[idx], (idx * patch_w, 0))

    # ---- plot ----
    fig = plt.figure(figsize=(15, 8))
    for i, (title, imgPIL) in enumerate([
        ("Original", orig_img),
        ("Soft Boundary Heatmap", soft_img),
        ("Hard Boundaries (Red Overlay)", hard_overlay_img),
    ]):
        ax = plt.subplot2grid((2, 3), (0, i))
        ax.imshow(imgPIL); ax.set_title(title); ax.axis("off")

    ax_raster = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    ax_raster.imshow(patch_strip)
    ax_raster.set_title("1D Rasterized Patches (Red = Boundary)")
    ax_raster.axis("off")

    import os
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":

    compression_rate = 0.25
    patch_size = 16
    checkpoint_type = "imagenet" # imagenet or CLIP
    tests = ["0", "1", "2", "3", "4", "5", "6", "7"]

    set_seed(42)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # converts to [0,1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    model = DTPViT(
        image_size=224,
        patch_size=patch_size,
        embed_dim=768,
        num_heads=12,
        depth=(4, 8, 0),
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

    ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/ImageNet_DRIP_78/model_299.pth"
    if checkpoint_type == "imagenet":
        model = load_backbone_from_imagenet_checkpoint(model, ckpt_path)

    elif checkpoint_type == "CLIP":
        model = load_dtpx_from_clip_checkpoint(model, ckpt_path)    
    model.eval()

    for test_index in tqdm(tests, desc="Visualizing Boundaries"):

        img_path = f"unit_inference_images/vis_test_{test_index}.jpg"
        img = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(img)

        visualize_boundaries_enhanced(
            model, 
            input_tensor,
            save_path=f"unit_visualization/boundary_visualization_{test_index}.png"
        )
