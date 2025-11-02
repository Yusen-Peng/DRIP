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

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, Optional, Tuple, List

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
def load_dtp_from_clip_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    map_location: str = "cpu",
    strict_head: bool = True,   # if False, exclude head.* from BOTH ckpt and model
    verbose: bool = True,
) -> Dict[str, object]:
    """
    STRICT loader for your new DTPViT from a CLIP-style checkpoint.

    Consumes only visual.* keys and remaps *conditionally* based on the model's state_dict:
      - patch_embed.* : keep 'patch_embed.proj.*' if the model has it; otherwise flatten to 'patch_embed.*'
      - boundary_predictor.* : keep nested 'boundary_predictor.boundary_predictor.*' if the model has it;
                               otherwise collapse one level to 'boundary_predictor.*'
      - visual.pos_emb -> pos_emb
      - visual.null_token -> null_token
      - pre_blocks/short_blocks/down_ln/head : strip the 'visual.' prefix only

    Zero tolerance: missing / unexpected / shape mismatch => RuntimeError.
    """

    # 0) Read checkpoint
    raw = torch.load(ckpt_path, map_location=map_location)
    sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(sd, dict):
        raise ValueError("Unsupported checkpoint format: expected a state_dict-like mapping.")

    # We'll need model keys to choose remap variants
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())

    expects_patch_proj = any(k.startswith("patch_embed.proj.") for k in model_keys)
    expects_patch_flat = any(k.startswith("patch_embed.weight") or k.startswith("patch_embed.bias") for k in model_keys)
    expects_bp_nested = any(k.startswith("boundary_predictor.boundary_predictor.") for k in model_keys)
    expects_bp_flat   = any(k.startswith("boundary_predictor.0.") or k.startswith("boundary_predictor.1.")
                            or k.startswith("boundary_predictor.2.") for k in model_keys)

    # 1) Keep only visual.* and strip prefix
    def strip_visual_prefix(k: str) -> Optional[str]:
        if k.startswith("module.visual."):
            return k[len("module.visual."):]
        if k.startswith("visual."):
            return k[len("visual."):]
        return None  # drop non-visual (e.g., text tower, globals)

    # 2) Minimal, conditional remap rules for the visual side
    def remap_visual(inner: str) -> Optional[str]:
        # ---- patch embed ----
        if inner.startswith("patch_embed.proj."):
            # keep 'proj.' form if the model expects it; otherwise flatten
            return inner if expects_patch_proj else inner.replace("patch_embed.proj.", "patch_embed.", 1)
        if inner.startswith("patch_embed.") and not inner.startswith("patch_embed.proj."):
            # if source is flat but model expects proj, expand is impossible; let strict check catch it
            return inner

        # ---- boundary predictor ----
        if inner.startswith("boundary_predictor.boundary_predictor."):
            # keep nested form if model expects it; otherwise collapse one level
            return (inner if expects_bp_nested else
                    inner.replace("boundary_predictor.boundary_predictor.", "boundary_predictor.", 1))
        if inner.startswith("boundary_predictor.") and not inner.startswith("boundary_predictor.boundary_predictor."):
            # if source is flat but model expects nested, we can't invent a level; let strict check catch it
            return inner

        # ---- pass-through (already aligned after stripping 'visual.') ----
        if inner.startswith(("pos_emb", "null_token",
                             "pre_blocks.", "short_blocks.",
                             "down_ln.", "head.")):
            return inner

        # Anything else under visual.* we keep as-is and let strict validation decide.
        return inner

    mapped: Dict[str, torch.Tensor] = OrderedDict()
    for k, v in sd.items():
        inner = strip_visual_prefix(k)
        if inner is None:
            continue
        mk = remap_visual(inner)
        if mk is None:
            continue
        if (not strict_head) and mk.startswith("head."):
            continue  # drop head from source if strict_head=False
        mapped[mk] = v

    # 3) Depth sanity: indices must match exactly
    def _idxs(prefix: str) -> List[int]:
        out = []
        pref = prefix + "."
        for k in mapped.keys():
            if k.startswith(pref):
                try:
                    out.append(int(k.split(".")[1]))
                except Exception:
                    pass
        return sorted(set(out))

    pre_idx  = _idxs("pre_blocks")
    short_idx = _idxs("short_blocks")
    exp_pre = list(range(len(getattr(model, "pre_blocks", []))))
    exp_short = list(range(len(getattr(model, "short_blocks", []))))

    if verbose:
        print(f"[drip] pre_blocks in ckpt: {pre_idx}  (expected {exp_pre})")
        print(f"[drip] short_blocks in ckpt: {short_idx}  (expected {exp_short})")

    if pre_idx != exp_pre or short_idx != exp_short:
        raise RuntimeError(
            f"[drip] Block index mismatch.\n"
            f"  ckpt pre_blocks:   {pre_idx} vs expected {exp_pre}\n"
            f"  ckpt short_blocks: {short_idx} vs expected {exp_short}"
        )

    # 4) Strict key/shape validation
    target_keys = set(model_keys)
    if not strict_head:
        target_keys = {k for k in target_keys if not k.startswith("head.")}

    src_keys = set(mapped.keys())

    missing = sorted(target_keys - src_keys)
    unexpected = sorted(src_keys - target_keys)

    shape_mismatch: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    for k in sorted(src_keys & target_keys):
        if tuple(model_sd[k].shape) != tuple(mapped[k].shape):
            shape_mismatch.append((k, tuple(mapped[k].shape), tuple(model_sd[k].shape)))

    if missing or unexpected or shape_mismatch:
        lines = ["[drip] Checkpoint does not exactly match the model."]
        if missing:
            lines.append(f"  - Missing in ckpt ({len(missing)}): {missing[:12]}{' ...' if len(missing)>12 else ''}")
        if unexpected:
            lines.append(f"  - Unexpected in ckpt ({len(unexpected)}): {unexpected[:12]}{' ...' if len(unexpected)>12 else ''}")
        if shape_mismatch:
            preview = [f"{k}: ckpt{ck} vs model{mk}" for k, ck, mk in shape_mismatch[:12]]
            lines.append(f"  - Shape mismatches ({len(shape_mismatch)}): " +
                         "; ".join(preview) + (" ..." if len(shape_mismatch)>12 else ""))
        raise RuntimeError("\n".join(lines))

    # 5) Load (strictness already enforced)
    model.load_state_dict({k: mapped[k] for k in sorted(src_keys)}, strict=False)

    if verbose:
        print(f"[drip] Loaded {len(src_keys)} tensors into DTPViT.")

    info = {
        "loaded": sorted(src_keys),
        "pre_blocks_indices_ckpt": pre_idx,
        "short_blocks_indices_ckpt": short_idx,
        "strict_head": strict_head,
    }

    return model, info


@torch.no_grad()
def load_vit_from_clip_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    map_location: str = "cpu",
    strict_proj: bool = True,     # set False to skip loading proj if shapes differ
    skip_head: bool = True,       # CLIP "head" is usually for classification; VisionTransformer doesn't use it
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Load ONLY the *vision transformer tower* from a CLIP-style checkpoint into VisionTransformer.

    Expected ckpt keys under visual.* (common OpenAI/CLIP/OpenCLIP patterns):
      - visual.conv1.{weight}
      - visual.class_embedding
      - visual.positional_embedding
      - visual.ln_pre.{weight,bias}
      - visual.transformer.resblocks.{i}.*   (attn, ln_*, mlp, etc.)
      - visual.ln_post.{weight,bias}
      - visual.proj                          (or visual.proj.{weight})
    Also handles legacy 'visual.patch_embed.*' -> model.conv1.* remap.

    This function:
      - strips 'visual.' / 'module.visual.' prefixes
      - remaps a few historical names (patch_embed -> conv1)
      - shape-checks everything; skips mismatches
      - returns a summary dict of what happened
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
    # 2) Remap legacy / alternate names to VisionTransformer's names
    # -------------------------------
    def remap_key(inner: str) -> str | None:
        # Map patch embed conv to conv1.*
        if inner.startswith("patch_embed.proj."):
            # e.g., patch_embed.proj.weight -> conv1.weight
            return inner.replace("patch_embed.proj.", "conv1.")
        if inner.startswith("patch_embed."):
            # e.g., patch_embed.weight -> conv1.weight
            return inner.replace("patch_embed.", "conv1.")

        # Some checkpoints store proj as a bare tensor "proj", others "proj.weight"
        if inner == "proj.weight":
            return "proj"  # our model registers proj as a Parameter (no .weight)
        # If it's already "proj" keep it as-is
        if inner == "proj":
            return inner

        # Everything else passes through (e.g., conv1.*, class_embedding,
        # positional_embedding, ln_pre.*, transformer.resblocks.*, ln_post.*)
        return inner

    visual_only = OrderedDict()
    for k, v in sd.items():
        inner = strip_visual_prefix(k)
        if inner is None:
            continue
        inner = remap_key(inner)
        if inner is None:
            continue
        # Optionally skip classification head if present in some variants
        if skip_head and inner.startswith("head."):
            continue
        visual_only[inner] = v

    # -------------------------------
    # 3) Collect indices for sanity (resblocks)
    # -------------------------------
    def _resblock_indices():
        out = []
        pref = "transformer.resblocks."
        for k in visual_only.keys():
            if k.startswith(pref):
                try:
                    idx = int(k.split(".")[2])
                    out.append(idx)
                except Exception:
                    pass
        return sorted(set(out))

    res_idx = _resblock_indices()

    # -------------------------------
    # 4) Shape-safe partial load
    # -------------------------------
    model_sd = model.state_dict()
    loadable = OrderedDict()
    skipped_shape: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    unexpected_in_ckpt: List[str] = []

    for k, v in visual_only.items():
        # Optionally require exact match for proj unless strict_proj=False
        if k == "proj" and k in model_sd and model_sd[k].shape != v.shape:
            if strict_proj:
                skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
                continue
            else:
                # Skip proj entirely if shapes differ and strict_proj is False
                continue

        if k not in model_sd:
            unexpected_in_ckpt.append(k)
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
        "resblock_indices_ckpt": res_idx,
    }

    if verbose:
        print(f"[vit] Loaded {len(loadable)} tensors into VisionTransformer.")
        if res_idx:
            print(f"[vit] transformer.resblocks in ckpt: {res_idx} (model has {len(getattr(model.transformer, 'resblocks', []))})")
        if skipped_shape:
            print(f"[vit] Skipped {len(skipped_shape)} (shape mismatch). First few:")
            for k, s_ckpt, s_model in skipped_shape[:8]:
                print(f"  - {k}: ckpt{s_ckpt} vs model{s_model}")
        if unexpected_in_ckpt:
            print(f"[vit] {len(unexpected_in_ckpt)} visual keys in ckpt not present in model.")
        if msg.missing_keys:
            print(f"[vit] {len(msg.missing_keys)} model keys missing in ckpt.")

    return model, info

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
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    axes = axes.flatten()

    for k in range(6):
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


def run_visualization(model, tests, preprocess, batch_size=8, out_dir="unit_visualization"):
    os.makedirs(out_dir, exist_ok=True)

    batch_tensors, batch_indices = [], []
    for test_index in tqdm(tests, desc="Visualizing Boundaries"):
        # load and preprocess
        img_path = f"unit_inference_images/vis_test_{test_index}.JPEG"
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

@torch.no_grad()
def visualize_boundaries_clean_noisy(
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
    names = ["clean_1", "noisy_1", "clean_2", "noisy_2"]
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

    plot_titles = ["clean_1", "noisy_1", "clean_2", "noisy_2"]

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



@torch.no_grad()
def visualize_boundaries_hard_soft_2x2(
    model: DTPViT,
    image_tensor_1: torch.Tensor,  # [3,H,W]
    image_tensor_2: torch.Tensor,  # [3,H,W]
    save_path: str | None = None,
):

    import cv2
    _use_cv2 = True
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # -------- helpers --------
    def denorm_to_pil(img_3chw: torch.Tensor) -> tuple[Image.Image, np.ndarray]:
        """Un-normalize ImageNet and return (PIL, np.uint8 array)."""
        orig = TF.normalize(
            img_3chw.clone(),
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225],
        ).clamp(0, 1).cpu()
        pil = TF.to_pil_image(orig).convert("RGB")
        return pil, np.array(pil).astype(np.uint8)

    def _one_image_overlays(img_3chw: torch.Tensor):
        """
        Run the DTPViT path for a single image and build:
          - hard_overlay_img (PIL): red patches where boundary=1
          - soft_img (PIL): heatmap
        """
        # ---- input ----
        x_img = img_3chw.unsqueeze(0).to(device)  # [1,3,H,W]

        # ---- patch embed -> [B, D, Hg, Wg] ----
        feat = model.patch_embed(x_img)
        B, D, Hg, Wg = feat.shape
        L = Hg * Wg

        # tokens [B, L, D]
        x = feat.flatten(2).transpose(1, 2).contiguous()

        # optional dropout / pos-drop
        drop = getattr(model, "dropout", None) or getattr(model, "pos_drop", None)
        if drop is not None:
            x = drop(x)

        # ---- positional encoding for pre_blocks ----
        # your pos_emb(seq) expects seq-first length L (no CLS)
        pos_seq = torch.arange(L-1, -1, -1.0, device=x.device, dtype=x.dtype)
        r = model.pos_emb(pos_seq)  # [L,1,D]

        # ---- run pre_blocks (seq-first) ----
        x = x.transpose(0, 1)  # [L, B, D]
        for block in model.pre_blocks:
            x = block(x, r, model.r_w_bias, model.r_r_bias)  # [L,B,D]

        # ---- boundary predictor ----
        soft_boundaries, hard_boundaries = model.boundary_predictor(x)  # likely [L,B] or [B,L]

        # normalize shapes to [B,L]
        if soft_boundaries.dim() == 2 and soft_boundaries.shape[0] == L:
            soft_boundaries = soft_boundaries.transpose(0, 1).contiguous()
            hard_boundaries = hard_boundaries.transpose(0, 1).contiguous()

        soft_mask = soft_boundaries[0].detach().float().cpu().view(Hg, Wg).numpy()
        hard_mask = hard_boundaries[0].detach().float().cpu().view(Hg, Wg).numpy()

        # ---- original image + dims ----
        orig_pil, orig_np = denorm_to_pil(img_3chw)
        img_size_attr = getattr(model, "img_size", None) or getattr(model, "image_size", None)
        image_size = int(img_size_attr) if img_size_attr is not None else orig_np.shape[0]

        patch_h = max(1, image_size // Hg)
        patch_w = max(1, image_size // Wg)

        # ---- soft heatmap (PIL) ----
        cmap = plt.get_cmap("hot")
        if _use_cv2:
            heat = cv2.resize(soft_mask, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        else:
            heat = np.array(Image.fromarray(soft_mask).resize((image_size, image_size), Image.BICUBIC))
        heat_colored = (cmap(heat)[..., :3] * 255).astype(np.uint8)
        soft_img = Image.fromarray(heat_colored)

        # ---- hard red overlay (PIL) ----
        # ensure base is exactly image_size x image_size for clean patches
        base_np = np.array(orig_pil.resize((image_size, image_size), Image.BICUBIC))
        red_overlay_np = base_np.copy()
        for i in range(Hg):
            for j in range(Wg):
                if hard_mask[i, j] > 0.5:
                    y0, y1 = i * patch_h, min((i + 1) * patch_h, image_size)
                    x0, x1 = j * patch_w, min((j + 1) * patch_w, image_size)
                    patch = red_overlay_np[y0:y1, x0:x1]
                    red = np.zeros_like(patch); red[..., 0] = 255
                    red_overlay_np[y0:y1, x0:x1] = (0.6 * patch + 0.4 * red).astype(np.uint8)
        hard_overlay_img = Image.fromarray(red_overlay_np)

        return hard_overlay_img, soft_img

    # -------- run both images --------
    hard1, soft1 = _one_image_overlays(image_tensor_1)
    hard2, soft2 = _one_image_overlays(image_tensor_2)

    # -------- plot 2x2 --------
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(hard1); axes[0, 0].set_title("Hard 1"); axes[0, 0].axis("off")
    axes[0, 1].imshow(soft1); axes[0, 1].set_title("Soft 1"); axes[0, 1].axis("off")
    axes[1, 0].imshow(hard2); axes[1, 0].set_title("Hard 2"); axes[1, 0].axis("off")
    axes[1, 1].imshow(soft2); axes[1, 1].set_title("Soft 2"); axes[1, 1].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")
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

    


    ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_4x_16_ViT_4_8/checkpoints/epoch_15.pt"
    load_dtp_from_clip_checkpoint(model, ckpt_path)
    ############### TESTING  ####################


    ############### TESTING  ####################
    # model = VisionTransformer(
    #         image_size=224,
    #         patch_size=patch_size,
    #         width=768,
    #         layers=12,
    #         heads=12,
    #         mlp_ratio=4.0
    # )
    # ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/ViT_B_16/checkpoints/epoch_15.pt"
    # load_vit_from_clip_checkpoint(model, ckpt_path)
    ############### TESTING  ####################



    
    
    # #ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_10x_16_XL_4_8/checkpoints/epoch_15.pt"
    # if checkpoint_type == "imagenet":
    #     model = load_backbone_from_imagenet_checkpoint(model, ckpt_path)

    # elif checkpoint_type == "CLIP":
    #     model, info = load_dtpx_from_clip_checkpoint(model, ckpt_path)    
    model.eval()

    # your list of test indices
    tests = ["0", "1", "2", "3", "4", "5"]

    # # run visualization (for the main paper)
    # # run_visualization(
    # #     model=model,
    # #     tests=tests,
    # #     preprocess=preprocess,
    # #     batch_size=6,   # 1x6 grid
    # #     out_dir="unit_visualization"
    # # )

    # # run 2x2 visualization (for the supplementary material)
    # # visualize_boundaries_single_multi(
    # #     model, 
    # #     preprocess=preprocess,
    # #     root_dir="/users/PAS2912/yusenpeng/Fast-CLIP/unit_further_vis/single_multi", 
    # #     save_path="/users/PAS2912/yusenpeng/Fast-CLIP/unit_further_vis/single_multi/boundary_visualization_2x2.png"
    # # )
    # # visualize_boundaries_clean_noisy(
    # #     model, 
    # #     preprocess=preprocess,
    # #     root_dir="/users/PAS2912/yusenpeng/Fast-CLIP/unit_further_vis/clean_noisy", 
    # #     save_path="/users/PAS2912/yusenpeng/Fast-CLIP/unit_further_vis/clean_noisy/boundary_visualization_2x2.png"
    # # )

    img1 = preprocess(Image.open("/users/PAS2912/yusenpeng/Fast-CLIP/unit_further_vis/soft_hard/img_1.JPEG").convert("RGB"))  # [3,H,W]
    img2 = preprocess(Image.open("/users/PAS2912/yusenpeng/Fast-CLIP/unit_further_vis/soft_hard/img_2.JPEG").convert("RGB"))  # [3,H,W]

    img1 = preprocess(Image.open("/users/PAS2912/yusenpeng/Fast-CLIP/unit_further_vis/soft_hard/img_3.JPEG").convert("RGB"))  # [3,H,W]
    img2 = preprocess(Image.open("/users/PAS2912/yusenpeng/Fast-CLIP/unit_further_vis/soft_hard/img_3.JPEG").convert("RGB"))  # [3,H,W]


    # visualize_boundaries_hard_soft_2x2(
    #     model,
    #     image_tensor_1=img1,
    #     image_tensor_2=img2,
    #     save_path="/users/PAS2912/yusenpeng/Fast-CLIP/unit_further_vis/soft_hard/boundary_visualization_2x2.png"
    # )

    visualize_boundaries_enhanced(
        model,
        image_tensor=torch.stack([img1]),
        save_path="/users/PAS2912/yusenpeng/Fast-CLIP/unit_further_vis/soft_hard/boundary_visualization_enhanced_TEST.png"
    )



