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


@torch.no_grad()
def weight_transfer(backbone: DTPViT):
    # 1) Load giver (OpenAI CLIP ViT-B/16) via open_clip
    clip_model, _, _ = create_model_and_transforms(
        model_name="ViT-B-16",
        pretrained="openai",
        DTP_ViT=False  # IMPORTANT: load the standard CLIP weights
    )
    clip_state_dict = clip_model.visual.state_dict()

    # Keep only vision-tower tensors we actually need
    clip_vit_state_dict = {
        k: v for k, v in clip_state_dict.items()
        if not k.startswith("proj")              # skip final projection
        and not k.startswith("ln_post")          # skip ln_post
        and "attn_mask" not in k                 # not a parameter
    }

    # 2) Receiver (your model)
    dtp_state_dict = backbone.state_dict()

    # --- Optional: quick previews you already printed ---
    # print("CLIP ViT-B/16 state dict keys:", flush=True)
    # print(clip_vit_state_dict.keys(), flush=True)
    # print("DTPViT state dict keys:", flush=True)
    # print(dtp_state_dict.keys(), flush=True)

    transferred, missing, mismatched = 0, [], []

    # 3) Patch embedding
    if "conv1.weight" not in clip_vit_state_dict:
        raise KeyError("Giver missing conv1.weight")
    if "patch_embed.proj.weight" not in dtp_state_dict:
        raise KeyError("Receiver missing patch_embed.proj.weight")

    if dtp_state_dict["patch_embed.proj.weight"].shape != clip_vit_state_dict["conv1.weight"].shape:
        mismatched.append(("patch_embed.proj.weight", tuple(dtp_state_dict["patch_embed.proj.weight"].shape),
                           "conv1.weight", tuple(clip_vit_state_dict["conv1.weight"].shape)))
    else:
        dtp_state_dict["patch_embed.proj.weight"].copy_(clip_vit_state_dict["conv1.weight"])
        if "patch_embed.proj.bias" in dtp_state_dict:
            dtp_state_dict["patch_embed.proj.bias"].zero_()
        transferred += 1
        print("✅ Transferred: patch_embed.proj (weight) and zeroed bias", flush=True)

    # 4) Positional embedding (CLS + patches)
    if "positional_embedding" not in clip_vit_state_dict:
        raise KeyError("Giver missing positional_embedding")
    if "pos_emb" not in dtp_state_dict:
        raise KeyError("Receiver missing pos_emb")


    src = clip_vit_state_dict["positional_embedding"]            # (197, 768)
    dst = dtp_state_dict["pos_emb"]                          # (1, 197, 768)

    if dst.shape[1:] != src.shape:                   # compare (197,768)
        # (You'd interpolate here if token count differs; not needed for 224/16.)
        raise ValueError(f"PosEmb token/dim mismatch: dst {dst.shape} vs src {src.shape}")

    dtp_state_dict["pos_emb"].copy_(src.unsqueeze(0))        # <- fix: add batch dim
    print("✅ Transferred: pos_emb (with unsqueeze(0))", flush=True)
    

    # 5) Blocks mapping
    # DTPViT uses:
    #   pre_blocks.{i}.(norm1/norm2, self_attn.in_proj_*, self_attn.out_proj.*, linear1/linear2)
    #   short_blocks.{j}.(...)
    n_pre = len(getattr(backbone, "pre_blocks", []))
    n_short = len(getattr(backbone, "short_blocks", []))
    total_needed = n_pre + n_short

    def dst_prefix_for_layer(i: int) -> str:
        if i < n_pre:
            return f"pre_blocks.{i}"
        else:
            j = i - n_pre
            return f"short_blocks.{j}"

    # Each OpenAI CLIP ViT-B/16 has 12 resblocks
    total_giver = 12
    max_layers = min(total_needed, total_giver)

    # Pairs: (giver_key_suffix, receiver_key_suffix)
    # CLIP:   ln_1 / ln_2, attn.in_proj_weight/bias, attn.out_proj.*, mlp.c_fc, mlp.c_proj
    # DTPViT: norm1/norm2, self_attn.in_proj_*    , self_attn.out_proj.*, linear1  , linear2
    pairs = [
        ("ln_1.weight",           "norm1.weight"),
        ("ln_1.bias",             "norm1.bias"),
        ("attn.in_proj_weight",   "self_attn.in_proj_weight"),
        ("attn.in_proj_bias",     "self_attn.in_proj_bias"),
        ("attn.out_proj.weight",  "self_attn.out_proj.weight"),
        ("attn.out_proj.bias",    "self_attn.out_proj.bias"),
        ("ln_2.weight",           "norm2.weight"),
        ("ln_2.bias",             "norm2.bias"),
        ("mlp.c_fc.weight",       "linear1.weight"),
        ("mlp.c_fc.bias",         "linear1.bias"),
        ("mlp.c_proj.weight",     "linear2.weight"),
        ("mlp.c_proj.bias",       "linear2.bias"),
    ]

    for i in range(max_layers):
        base = f"transformer.resblocks.{i}"
        dst_prefix = dst_prefix_for_layer(i)

        # sanity: receiver has this block?
        needs = [f"{dst_prefix}.{r}" for _, r in pairs]
        has_block = any(k.startswith(f"{dst_prefix}.") for k in dtp_state_dict.keys())
        if not has_block:
            missing.append((dst_prefix, "entire block missing in receiver"))
            print(f"⚠️ Skipping {dst_prefix}: block not found in receiver state_dict.", flush=True)
            continue

        copied_this_block = 0
        for g_suf, r_suf in pairs:
            g_key = f"{base}.{g_suf}"
            r_key = f"{dst_prefix}.{r_suf}"
            g_val = clip_vit_state_dict.get(g_key, None)
            r_val = dtp_state_dict.get(r_key, None)

            if g_val is None or r_val is None:
                missing.append((g_key if g_val is None else r_key, "missing"))
                continue

            if r_val.shape != g_val.shape:
                mismatched.append((r_key, tuple(r_val.shape), g_key, tuple(g_val.shape)))
                continue

            r_val.copy_(g_val)
            transferred += 1
            copied_this_block += 1

        print(f"✅ Transferred block {i:02d} → {dst_prefix}  (copied {copied_this_block}/{len(pairs)})", flush=True)

    # 6) Commit into model (allow leftovers that DTPViT owns)
    backbone.load_state_dict(dtp_state_dict, strict=False)

    # 7) Summary
    print("\n=== Transfer Summary ===", flush=True)
    print(f"Needed layers (receiver): {total_needed}  |  Giver layers available: {total_giver}  |  Mapped: {max_layers}", flush=True)
    print(f"Transferred tensors: {transferred}", flush=True)
    if missing:
        print(f"Missing ({len(missing)}): first few -> {missing[:8]}", flush=True)
    if mismatched:
        print(f"Mismatched ({len(mismatched)}): first few -> {mismatched[:8]}", flush=True)
    print("========================\n", flush=True)




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

# @torch.no_grad()
# def load_dtpx_from_clip_checkpoint(model: nn.Module, ckpt_path: str) -> DTPViT:
#     """
#     Loads weights into a DTPViT model from a CLIP-style checkpoint.

#     Args:
#         model (DTPViT): An uninitialized DTPViT model with the correct config.
#         ckpt_path (str): Path to a checkpoint with 'module.visual.' prefix in keys.

#     Returns:
#         model (DTPViT): The same model with loaded weights.
#     """
#     ckpt = torch.load(ckpt_path, map_location='cpu')
#     raw_state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

#     print("Keys in checkpoint:")
#     for k in raw_state_dict.keys():
#         print(k)

#     dtpvit_state_dict = {
#         k.replace("module.visual.", ""): v
#         for k, v in raw_state_dict.items()
#         if k.startswith("module.visual.")
#     }

#     #model.load_state_dict(dtpvit_state_dict, strict=False)

#     # check if model is loaded correctly
#     missing_keys, unexpected_keys = model.load_state_dict(dtpvit_state_dict, strict=False)
#     print(f"[CLIP→DTPViT] missing={len(missing_keys)}  unexpected={len(unexpected_keys)}")
#     if missing_keys:    print("  missing (first 15):", missing_keys[:15])

#     return model


import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, Tuple, List

@torch.no_grad()
def load_dtpx_from_clip_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    map_location: str = "cpu",
    strict_head: bool = True,   # set False if num_classes differs or you want to re-init the head
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Load ONLY the vision encoder (visual.*) from a CLIP-style checkpoint into DTPViT.

    - Accepts checkpoints with top-level keys like:
        visual.r_w_bias, visual.r_r_bias, visual.null_token, visual.patch_embed.{weight,bias},
        visual.pos_emb.inv_freq, visual.pre_blocks.{i}.*, visual.short_blocks.{j}.*,
        visual.boundary_predictor.boundary_predictor.{0,2}.{weight,bias},
        visual.down_ln.{weight,bias}, visual.head.{weight,bias}
      (and ignores text/global keys like positional_embedding, text_projection, logit_scale, transformer.*)

    - Safe partial load: skips shape-mismatched tensors, logs what happened.
    - Minimal remapping hook for historical names (extend in `remap_key` if needed).
    """
    # -------------------------------
    # 0) Load raw state dict
    # -------------------------------
    raw = torch.load(ckpt_path, map_location=map_location)
    sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(sd, dict):
        raise ValueError("Unsupported checkpoint format: expected a state_dict-like dict.")

    # -------------------------------
    # 1) Keep only visual.* keys, strip prefix
    # -------------------------------
    def strip_visual_prefix(k: str):
        if k.startswith("module.visual."):
            return k[len("module.visual."):]
        if k.startswith("visual."):
            return k[len("visual."):]
        return None

    # -------------------------------
    # 2) (Optional) remap legacy names
    # -------------------------------
    def remap_key(k: str) -> str:
        # Example legacy: "patch_embed.proj.weight" -> "patch_embed.weight"
        if k.startswith("patch_embed.proj."):
            k = k.replace("patch_embed.proj.", "patch_embed.")
        # If you ever renamed boundary predictor internals, add rules here.
        return k

    visual_only = OrderedDict()
    for k, v in sd.items():
        inner = strip_visual_prefix(k)
        if inner is None:
            continue
        inner = remap_key(inner)
        if not strict_head and inner.startswith("head."):
            continue  # skip classifier head if requested
        visual_only[inner] = v

    # -------------------------------
    # 3) Quick depth sanity check (optional but helpful)
    # -------------------------------
    def _indices_present(prefix: str) -> List[int]:
        out = []
        pref = f"{prefix}."
        for k in visual_only.keys():
            if k.startswith(pref):
                # e.g., "pre_blocks.3.dec_attn.qkv_net.weight" -> 3
                try:
                    idx = int(k.split(".")[1])
                    out.append(idx)
                except Exception:
                    pass
        return sorted(set(out))

    pre_idx  = _indices_present("pre_blocks")
    short_idx = _indices_present("short_blocks")

    # Your model has depth=(4,8,0): pre 0..3, short 0..7
    expected_pre = set(range(len(getattr(model, "pre_blocks", []))))
    expected_short = set(range(len(getattr(model, "short_blocks", []))))

    if verbose:
        print(f"[drip] pre_blocks in ckpt: {pre_idx}  (expected {sorted(expected_pre)})")
        print(f"[drip] short_blocks in ckpt: {short_idx}  (expected {sorted(expected_short)})")

    # -------------------------------
    # 4) Shape-safe partial load
    # -------------------------------
    model_sd = model.state_dict()
    loadable = OrderedDict()
    skipped_shape: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    unexpected_in_ckpt: List[str] = []

    for k, v in visual_only.items():
        if k not in model_sd:
            unexpected_in_ckpt.append(k)  # exists in ckpt but no matching param/buffer in model
            continue
        if model_sd[k].shape != v.shape:
            skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        loadable[k] = v

    msg = model.load_state_dict(loadable, strict=False)

    info = {
        "loaded": sorted(loadable.keys()),
        "skipped_shape_mismatch": skipped_shape,                 # list of (key, ckpt_shape, model_shape)
        "unexpected_in_checkpoint": sorted(unexpected_in_ckpt),  # extra ckpt tensors not used
        "missing_keys_reported": sorted(msg.missing_keys),       # model tensors not found in ckpt
        "unexpected_keys_reported": sorted(msg.unexpected_keys), # should be empty
        "pre_blocks_indices_ckpt": pre_idx,
        "short_blocks_indices_ckpt": short_idx,
    }

    if verbose:
        print(f"[drip] Loaded {len(loadable)} tensors into DTPViT.")
        if skipped_shape:
            print(f"[drip] Skipped {len(skipped_shape)} (shape mismatch). First few:")
            for k, s_ckpt, s_model in skipped_shape[:8]:
                print(f"  - {k}: ckpt{s_ckpt} vs model{s_model}")
        if unexpected_in_ckpt:
            print(f"[drip] {len(unexpected_in_ckpt)} visual keys in ckpt not present in model (config/depth mismatch likely).")
        if msg.missing_keys:
            print(f"[drip] {len(msg.missing_keys)} model keys missing in ckpt (newer modules or different config).")

    return model, info
    
    


@torch.no_grad()
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
    import os

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- normalize input shape to [N,3,H,W] ---
    if image_tensor.dim() == 3:   # [3,H,W]
        batch = image_tensor.unsqueeze(0)
    elif image_tensor.dim() == 4: # [N,3,H,W]
        batch = image_tensor
    else:
        raise ValueError(f"image_tensor must be [3,H,W] or [N,3,H,W], got {tuple(image_tensor.shape)}")

    batch = batch.to(device)
    N = batch.size(0)

    # --- helper: build hard-overlay image for a single item ---
    def overlay_hard(img_3chw: torch.Tensor) -> Image.Image:
        # patch embed
        feat = model.patch_embed(img_3chw.unsqueeze(0))     # [1, D, Hg, Wg]
        _, D, Hg, Wg = feat.shape
        grid_h, grid_w = Hg, Wg

        # tokens [B=1, L, D]
        x = feat.flatten(2).transpose(1, 2).contiguous()

        # optional drop
        drop = getattr(model, "dropout", None) or getattr(model, "pos_drop", None)
        if drop is not None:
            x = drop(x)

        # positional encoding (no CLS assumed)
        L = x.size(1)
        pos_seq = torch.arange(L - 1, -1, -1.0, device=x.device, dtype=x.dtype)
        r = model.pos_emb(pos_seq)  # [L,1,D]

        # pre_blocks expect [L,B,D]
        x = x.transpose(0, 1)  # [L,1,D]
        for block in getattr(model, "pre_blocks", []):
            x = block(x, r, model.r_w_bias, model.r_r_bias)  # [L,1,D]

        # boundary predictor
        _, hard_boundaries = model.boundary_predictor(x)  # shapes vary

        # ensure [B,L]
        if hard_boundaries.dim() == 2 and hard_boundaries.shape[0] == L:  # [L,B]
            hard_boundaries = hard_boundaries.transpose(0, 1).contiguous()  # [B,L]

        hard_mask = hard_boundaries[0].detach().float().view(grid_h, grid_w).cpu().numpy()

        # recover original (undo ImageNet norm if applied)
        orig = TF.normalize(
            img_3chw.detach().clone().cpu(),
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        ).clamp(0, 1)
        orig_img = TF.to_pil_image(orig).convert("RGB")
        orig_np = np.array(orig_img).astype(np.uint8)

        image_size = getattr(model, "image_size", orig_np.shape[0])
        # If the model’s image_size differs from current image, resize for clean tiling
        if orig_np.shape[0] != image_size or orig_np.shape[1] != image_size:
            orig_img = orig_img.resize((image_size, image_size), Image.BILINEAR)
            orig_np = np.array(orig_img).astype(np.uint8)

        patch_h = image_size // grid_h
        patch_w = image_size // grid_w

        # red overlay where hard_mask = 1
        red_overlay_np = orig_np.copy()
        for i in range(grid_h):
            for j in range(grid_w):
                if hard_mask[i, j] == 1.:
                    y0, y1 = i * patch_h, (i + 1) * patch_h
                    x0, x1 = j * patch_w, (j + 1) * patch_w
                    patch = red_overlay_np[y0:y1, x0:x1]
                    red = np.zeros_like(patch); red[..., 0] = 255
                    red_overlay_np[y0:y1, x0:x1] = (0.6 * patch + 0.4 * red).astype(np.uint8)

        return Image.fromarray(red_overlay_np)

    max_show = min(N, 6)
    overlays = [overlay_hard(batch[i]) for i in range(max_show)]

    # --- plot in 1×6 grid (fill blanks if fewer than 6) ---
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes = axes.flatten()

    for k in range(2):
        ax = axes[k]
        ax.axis("off")
        if k < max_show:
            ax.imshow(overlays[k])
            if k == 0:
                ax.set_title("shark")
            elif k == 1:
                ax.set_title("dog")
            elif k == 2:
                ax.set_title("parachute")
            elif k == 3:
                ax.set_title("phone")
            elif k == 4:
                ax.set_title("goat")
            elif k == 5:
                ax.set_title("cat")
            

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def run_visualization(model, tests, preprocess, batch_size=2, out_dir="unit_visualization"):
    os.makedirs(out_dir, exist_ok=True)

    batch_tensors, batch_indices = [], []
    for test_index in tqdm(tests, desc="Visualizing Boundaries"):
        # load and preprocess
        img_path = f"unit_further_vis/mmbench_image/mmbench_{test_index}.png"
        img = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(img)  # [3,H,W]

        batch_tensors.append(input_tensor)
        batch_indices.append(test_index)

    # when batch is full → visualize and save
    if len(batch_tensors) == batch_size:
        save_path = os.path.join(out_dir,
            f"boundary_visualization_{batch_indices[0]}-{batch_indices[-1]}.png")
        visualize_boundaries_enhanced(model, torch.stack(batch_tensors), save_path=save_path)
        batch_tensors, batch_indices = [], []

@torch.no_grad()
def visualize_boundaries_single_multi(
    model: DTPViT, 
    preprocess,
    root_dir: str = ".", 
    save_path: str = None):
    """
    Reads four images:
        root_dir/single_1.JPEG
        root_dir/multi_1.JPEG
        root_dir/single_2.JPEG
        root_dir/multi_2.JPEG
    and produces a 2x2 boundary visualization figure:

      [ single_1 | multi_1 ]
      [ Single_2 | multi_2 ]

    If save_path is provided, saves the figure there; otherwise shows it.
    """
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision.transforms.functional as TF

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    # ---------- helper: load & preprocess (ImageNet norm) ----------
    def load_img_norm(path: str, preprocess) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        tensor = preprocess(img)  # [3,H,W]
        return tensor

    # ---------- helper: single-image overlay (same logic as yours) ----------
    def overlay_hard(img_3chw_norm: torch.Tensor) -> Image.Image:
        # patch embed
        feat = model.patch_embed(img_3chw_norm.unsqueeze(0).to(device))  # [1, D, Hg, Wg]
        _, D, Hg, Wg = feat.shape
        grid_h, grid_w = Hg, Wg

        # tokens [B=1, L, D]
        x = feat.flatten(2).transpose(1, 2).contiguous()  # [1, L, D]

        # optional drop
        drop = getattr(model, "dropout", None) or getattr(model, "pos_drop", None)
        if drop is not None:
            x = drop(x)

        # positional enc (no CLS)
        L = x.size(1)
        pos_seq = torch.arange(L - 1, -1, -1.0, device=x.device, dtype=x.dtype)
        r = model.pos_emb(pos_seq)  # [L,1,D]

        # pre_blocks expect [L,B,D]
        x = x.transpose(0, 1)  # [L,1,D]
        for block in getattr(model, "pre_blocks", []):
            x = block(x, r, model.r_w_bias, model.r_r_bias)  # [L,1,D]

        # boundary predictor
        _, hard_boundaries = model.boundary_predictor(x)  # shapes vary

        # ensure [B,L]
        if hard_boundaries.dim() == 2 and hard_boundaries.shape[0] == L:  # [L,B]
            hard_boundaries = hard_boundaries.transpose(0, 1).contiguous()  # [B,L]

        hard_mask = hard_boundaries[0].detach().float().view(grid_h, grid_w).cpu().numpy()

        # reconstruct display image (undo ImageNet norm)
        disp = TF.normalize(
            img_3chw_norm.detach().clone().cpu(),
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        ).clamp(0, 1)
        orig_img = TF.to_pil_image(disp).convert("RGB")
        orig_np = np.array(orig_img).astype(np.uint8)

        image_size = getattr(model, "image_size", orig_np.shape[0])
        if orig_np.shape[0] != image_size or orig_np.shape[1] != image_size:
            orig_img = orig_img.resize((image_size, image_size), Image.BILINEAR)
            orig_np = np.array(orig_img).astype(np.uint8)

        patch_h = image_size // grid_h
        patch_w = image_size // grid_w

        red_overlay_np = orig_np.copy()
        for i in range(grid_h):
            for j in range(grid_w):
                if hard_mask[i, j] == 1.0:
                    y0, y1 = i * patch_h, (i + 1) * patch_h
                    x0, x1 = j * patch_w, (j + 1) * patch_w
                    patch = red_overlay_np[y0:y1, x0:x1]
                    red = np.zeros_like(patch); red[..., 0] = 255
                    red_overlay_np[y0:y1, x0:x1] = (0.6 * patch + 0.4 * red).astype(np.uint8)

        return Image.fromarray(red_overlay_np)

    # ---------- read the four files ----------
    names = ["single_1", "multi_1", "single_2", "multi_2"]
    paths = [os.path.join(root_dir, f"{n}.JPEG") for n in names]

    tensors = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        tensors.append(load_img_norm(p, preprocess))

    # ---------- build overlays ----------
    overlays = [overlay_hard(t) for t in tensors]

    # ---------- plot 2x2 ----------
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    # titles as requested (note the capital 'S' in 'Single_2')
    plot_titles = ["single_1", "multi_1", "Single_2", "multi_2"]

    for ax, ov, title in zip(axes, overlays, plot_titles):
        ax.imshow(ov)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


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

    #ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_10x_16_XL_4_8/checkpoints/epoch_15.pt"
    ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_4x_16_XL_4_8/checkpoints/epoch_15.pt"

    #ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/ImageNet_DRIP_78/model_299.pth"
    if checkpoint_type == "imagenet":
        model = load_backbone_from_imagenet_checkpoint(model, ckpt_path)

    elif checkpoint_type == "CLIP":
        model, info = load_dtpx_from_clip_checkpoint(model, ckpt_path)    
    model.eval()

    # your list of test indices
    #tests = ["0", "1", "2", "3", "4", "5"]
    # tests = ["0", "1"]
    # tests = ["2", "3"]
    tests = ["8", "9"]

    # run visualization (for the main paper)
    # run_visualization(
    #     model=model,
    #     tests=tests,
    #     preprocess=preprocess,
    #     batch_size=2,   # 1x6 grid
    #     out_dir="unit_further_vis/mmbench_image"
    # )

    # run 2x2 visualization (for the supplementary material)
    visualize_boundaries_single_multi(
        model, 
        preprocess=preprocess,
        root_dir="/users/PAS2912/yusenpeng/Fast-CLIP/unit_further_vis/single_multi", 
        save_path="/users/PAS2912/yusenpeng/Fast-CLIP/unit_further_vis/single_multi/TransformerXL_based_boundary_visualization_2x2_4x.png"
    )
