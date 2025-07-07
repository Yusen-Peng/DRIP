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
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_dtpx_from_clip_checkpoint(model: nn.Module, ckpt_path: str):
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
    return model

@torch.no_grad()
def visualize_boundaries(model, image_tensor, save_path=None):
    """
    Visualizes learned boundaries over the input image side-by-side.

    Args:
        model: trained DTPViT model
        image_tensor: torch.Tensor of shape [3, H, W] (should be preprocessed)
        save_path: if given, saves the visualization
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img = image_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]

    # Patch embedding
    patches = model.patch_embed(img)  # [1, N, D]
    B, N, D = patches.shape

    # Add CLS token
    cls_token = model.cls_token.expand(B, -1, -1)
    x = torch.cat([cls_token, patches], dim=1)
    x = x + model.pos_embed[:, :x.size(1), :]
    x = model.pos_drop(x)
    x = x.transpose(0, 1)  # [1+N, B, D]
    x = model.pre_blocks(x)

    # boundary prediction
    _, hard_boundaries = model.boundary_predictor(x)

    # remove CLS token
    hard_boundaries = hard_boundaries[:, 1:]  # [B, N]

    # Get boundary map (binary mask)
    boundary_mask = hard_boundaries[0].cpu().view(
        model.patch_embed.grid_size[0],
        model.patch_embed.grid_size[1]
    ).numpy()

    # Upsample boundary mask to image size
    mask_img = TF.resize(
        TF.to_pil_image(torch.tensor(boundary_mask).unsqueeze(0)),
        size=(model.image_size, model.image_size),
        interpolation=TF.InterpolationMode.NEAREST
    )

    # Convert image tensor back to displayable image
    orig_img = TF.normalize(image_tensor.clone(),
                            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    orig_img = TF.to_pil_image(orig_img.cpu().clamp(0, 1))

    # Overlay version
    overlay_img = orig_img.copy()
    overlay_img.paste(mask_img.convert("RGB"), (0, 0), mask_img.convert("L"))

    # Plot both
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(orig_img)
    axs[0].set_title("Original")
    axs[0].axis('off')

    axs[1].imshow(overlay_img)
    axs[1].set_title("Boundaries Annotated")
    axs[1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":

    test_index = 1
    compression = "10x"  # or "4x", "10x"


    set_seed(42)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # converts to [0,1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_path = f"uni_test/vis_test_{test_index}.jpg"
    img = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(img)  # [3, 224, 224]


    model = DTPViT(
        image_size=224,
        patch_size=32,
        embed_dim=768,
        num_heads=12,
        depth=(2, 10, 0),
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.1,
        num_classes=512,
        temp=0.5,
        compression_rate=0.1,
        threshold=0.5,
        activation_function='gelu',
        flop_measure=False
    )


    ckpt_path = f"logs/DTP-ViT-{compression}-32/checkpoints/epoch_10.pt"
    model = load_dtpx_from_clip_checkpoint(model, ckpt_path)    
    model.eval()
    visualize_boundaries(
        model, 
        input_tensor,
        save_path=f"boundary_visualization_{test_index}_{compression}.png"
    )
