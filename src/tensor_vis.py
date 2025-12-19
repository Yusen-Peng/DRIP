import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from open_clip_local import create_model_and_transforms
from open_clip_local.model import DTPViT
from open_clip_local.transformer import VisionTransformer
from open_clip_local import CLIP
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from collections import OrderedDict
import os
import math
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
torch.serialization.add_safe_globals([argparse.Namespace])


from open_clip_local.model import DTPViT, VisionTransformer
from boundary_vis_dev import load_dtp_from_clip_checkpoint, set_seed, load_vit_from_clip_checkpoint


def main(model: DTPViT | VisionTransformer):
    model.eval()
    with torch.no_grad():
        pos_emb = model.positional_embedding.detach().cpu().float()  # [L+1, D]
    cls_pos = pos_emb[0] # [D]
    patch_pos = pos_emb[1:] # [L, D]
    gh, gw = model.grid_size
    assert patch_pos.shape[0] == gh * gw, "Grid size mismatch"

    print(f"[pos emb] cls shape: {cls_pos.shape}")
    print(f"[pos emb] patch shape: {patch_pos.shape}")
    print(f"[pos emb] mean={patch_pos.mean():.4f}, std={patch_pos.std():.4f}")

    # grid coordinates (x, y)
    coords = np.stack(np.meshgrid(np.arange(gw), np.arange(gh)), axis=-1).reshape(-1, 2)   # [L, 2]

    # t-SNE visualization (global geometry)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=0
    )

    Z = tsne.fit_transform(patch_pos.numpy())  # [L, 2]

    plt.figure(figsize=(6, 5))
    plt.scatter(
        Z[:, 0],
        Z[:, 1],
        c=coords[:, 1],      # color by row (y)
        s=12,
        cmap="viridis"
    )
    plt.colorbar(label="row index (y)")
    plt.title("t-SNE of Positional Embeddings (patch tokens)")
    plt.tight_layout()
    plt.show()
    plt.savefig("tsne/tsne_pos_emb.png")

    return


if __name__ == "__main__":
    compression_rate = 0.25
    patch_size = 16
    checkpoint_type = "CLIP" # imagenet or CLIP

    set_seed(42)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # converts to [0,1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    MODEL_TYPE = "ViT" # "ViT" or "DRIP"

    if MODEL_TYPE == "ViT":
        model_empty = VisionTransformer(
            image_size=224,
            patch_size=patch_size,
            width=768,
            layers=12,
            heads=768 // 64,
            mlp_ratio=4.0,
            pos_embed_type="learnable",  # "sin_cos_2d" or "learnable"
        )

        ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/ViT_B_16/checkpoints/epoch_15.pt"
        model, _ = load_vit_from_clip_checkpoint(model_empty, ckpt_path)


    elif MODEL_TYPE == "DRIP":
        model_empty = DTPViT(
            image_size=224,
            patch_size=patch_size,
            width=768,
            layers=12,
            depth=(4, 8, 0),
            compression_rate=compression_rate,
            heads=768 // 64,
            mlp_ratio=4.0,
            temp=0.5,
            pos_embed_type="sin_cos_2d",  # "sin_cos_2d" or "learnable"
            flop_measure=False # need to learn real boundaries
        )

        #ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/faithful_DRIP_4x_4_8_1e-3/checkpoints/epoch_4.pt"
        #ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/faithful_DRIP_4x_4_8_5e-5/checkpoints/epoch_4.pt"
        #ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/faithful_DRIP_4x_4_8_1e-4/checkpoints/epoch_4.pt"
        ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/faithful_DRIP_sinusoidal/checkpoints/epoch_4.pt"
        #ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_4x_16_ViT_4_8_NEW/checkpoints/epoch_2.pt"
        #ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/faithful_vitbased_DRIP_4epoch/checkpoints/epoch_4.pt"
        #ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/vitbased_drip_4epochs_sinusoidal_built_in_nn/checkpoints/epoch_4.pt"
        #ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/faithful_vitbased_drip_4epochs_sinusoidal/checkpoints/epoch_4.pt"
        #ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_4x_16_ViT_4_8/checkpoints/epoch_15.pt"
        #ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_10x_16_ViT_4_8/checkpoints/epoch_15.pt"
        model, _ = load_dtp_from_clip_checkpoint(model_empty, ckpt_path)


    main(model)
