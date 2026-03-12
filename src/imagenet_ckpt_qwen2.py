import os
from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
from open_clip_local.Qwen2VL_ViT import Qwen2VLVisionConfig, Qwen2VLDRIP, Qwen2VLVisionBlock, apply_rotary_pos_emb_vision

def load_img_norm(img_path, preprocess):
    """
    Returns a normalized tensor [3, H, W].
    Assumes preprocess returns a tensor ready for the model.
    """
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img)
    if isinstance(x, dict):
        # just in case preprocess is HF-style and returns dict
        if "pixel_values" in x:
            x = x["pixel_values"]
            if x.ndim == 4:
                x = x[0]
        else:
            raise ValueError("Unsupported preprocess dict format.")
    return x

def unnormalize_img(img_3chw, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Convert normalized tensor back to displayable [0,1] RGB tensor.
    """
    mean = list(mean)
    std = list(std)
    return TF.normalize(
        img_3chw.detach().clone().cpu(),
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1.0 / s for s in std],
    ).clamp(0, 1)

def _extract_state_dict(ckpt):
    """
    Be forgiving about checkpoint structure.
    """
    candidate_keys = [
        "state_dict",
        "model",
        "model_state_dict",
        "module",
        "network",
    ]
    if isinstance(ckpt, dict):
        for k in candidate_keys:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # maybe the checkpoint itself is already a state_dict
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError("Could not find a usable state_dict in checkpoint.")

def _strip_prefix_if_needed(state_dict, prefixes=("module.", "visual.", "backbone.")):
    out = {}
    for k, v in state_dict.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    changed = True
        out[nk] = v
    return out

def load_qwen2vldrip_checkpoint(model: Qwen2VLDRIP, checkpoint_path, device="cpu", strict=False, verbose=True):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(ckpt)
    state_dict = _strip_prefix_if_needed(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    model.to(device).eval()

    if verbose:
        print(f"[load_qwen2vldrip_checkpoint] loaded from: {checkpoint_path}")
        print(f"  missing keys: {len(missing)}")
        print(f"  unexpected keys: {len(unexpected)}")
        if len(missing) and len(missing) < 50:
            print("  missing:", missing)
        if len(unexpected) and len(unexpected) < 50:
            print("  unexpected:", unexpected)

    return model

@torch.no_grad()
def get_qwen2vldrip_hard_boundaries(model: Qwen2VLDRIP, img_3chw: torch.Tensor):
    """
    For Qwen2VLDRIP:
      input image -> patch_embed -> pre blocks -> boundary_predictor
    Returns:
      hard_boundaries: [L]
      grid_h, grid_w
    """
    device = next(model.parameters()).device
    x = img_3chw.unsqueeze(0).to(device)   # [1, 3, H, W]

    B, _, H, W = x.shape
    gh = H // model.config.patch_size
    gw = W // model.config.patch_size

    grid_thw = x.new_empty((B, 3), dtype=torch.long)
    grid_thw[:, 0] = 1
    grid_thw[:, 1] = gh
    grid_thw[:, 2] = gw

    # patch embedding
    hidden_states = model.patch_embed(x)   # [B*L, D] since B=1, effectively [L, D]

    # rotary embeddings
    rotary_pos_emb = model.rot_pos_emb(grid_thw)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # pre blocks
    for blk in model.blocks_pre:
        hidden_states: torch.Tensor = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )

    # [L, D] -> [1, L, D]
    reshaped_hidden_states = hidden_states.view(B, gh * gw, model.config.embed_dim)

    if model.flop_measure:
        # deterministic fallback if flop_measure=True
        L = reshaped_hidden_states.shape[1]
        num_tokens_to_keep = max(1, int(L * model.prior))
        indices = torch.linspace(0, L - 1, steps=num_tokens_to_keep, device=device).round().long()
        hard_boundaries = torch.zeros(B, L, device=device)
        hard_boundaries[:, indices] = 1
    else:
        x_transposed = reshaped_hidden_states.transpose(0, 1)  # [L, B, D]
        """
            NOTE: for visualization/inference, we stop sampling and apply deterministic thresholding
        """
        # _, hard_boundaries = model.boundary_predictor(x_transposed)  # [B, L]
        boundary_logits = model.boundary_predictor.boundary_predictor(x_transposed).squeeze(-1).transpose(0, 1)   # [B, L]
        boundary_probs = torch.sigmoid(boundary_logits)
        hard_boundaries = (boundary_probs > model.boundary_predictor.threshold).float()

    return hard_boundaries[0].detach(), gh, gw


@torch.no_grad()
def overlay_qwen2vldrip_boundaries(
    model: Qwen2VLDRIP,
    img_3chw: torch.Tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """
    Returns:
      PIL.Image with boundary-kept patches overlaid in red.
    """
    hard_1d, grid_h, grid_w = get_qwen2vldrip_hard_boundaries(model, img_3chw)

    assert hard_1d.numel() == grid_h * grid_w, \
        f"Expected {grid_h * grid_w} patch tokens, got {hard_1d.numel()}"

    hard_mask = hard_1d.float().view(grid_h, grid_w).cpu().numpy()

    print("hard boundaries shape:", hard_1d.shape)
    print("hard mask:")
    print(hard_mask)

    # unnormalize image for visualization
    orig = unnormalize_img(img_3chw, mean=mean, std=std)
    orig_img = TF.to_pil_image(orig).convert("RGB")
    orig_np = np.array(orig_img).astype(np.uint8)

    img_h, img_w = orig_np.shape[0], orig_np.shape[1]
    patch_h = img_h // grid_h
    patch_w = img_w // grid_w

    red_overlay_np = orig_np.copy()
    for i in range(grid_h):
        for j in range(grid_w):
            if hard_mask[i, j] == 1.0:
                y0, y1 = i * patch_h, (i + 1) * patch_h
                x0, x1 = j * patch_w, (j + 1) * patch_w
                patch = red_overlay_np[y0:y1, x0:x1]
                red = np.zeros_like(patch)
                red[..., 0] = 255
                red_overlay_np[y0:y1, x0:x1] = (0.6 * patch + 0.4 * red).astype(np.uint8)

    return Image.fromarray(red_overlay_np)

@torch.no_grad()
def visualize_boundaries_single_multi_qwen2vldrip(
    model: Qwen2VLDRIP,
    preprocess,
    root_dir="/users/PAS2912/yusenpeng/Fast-CLIP/src/boundary_vis/image_samples/single_multi/",
    save_path="/users/PAS2912/yusenpeng/Fast-CLIP/src/boundary_vis/single_multi_overlay.png",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """
    Reads:
      single_1.JPEG, multi_1.JPEG, single_2.JPEG, multi_2.JPEG
    and plots boundary overlays in a 2x2 grid.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    names = ["single_1", "multi_1", "single_2", "multi_2"]
    paths = [os.path.join(root_dir, f"{n}.JPEG") for n in names]

    tensors = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        tensors.append(load_img_norm(p, preprocess))

    overlays = [
        overlay_qwen2vldrip_boundaries(model, t, mean=mean, std=std)
        for t in tensors
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    plot_titles = ["single_1", "multi_1", "Single_2", "multi_2"]

    for ax, ov, title in zip(axes, overlays, plot_titles):
        ax.imshow(ov)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


@torch.no_grad()
def get_qwen2vldrip_layer4_attention_no_patch(
    model: Qwen2VLDRIP,
    img_3chw: torch.Tensor,
):
    """
    Compute attention map from blocks_pre[3] without modifying model code.
    Returns:
        attn_map: [gh, gw] normalized patch heatmap
        gh, gw
    """
    device = next(model.parameters()).device
    x = img_3chw.unsqueeze(0).to(device)  # [1, 3, H, W]

    B, _, H, W = x.shape
    assert B == 1, "This helper currently assumes a single image."
    gh = H // model.config.patch_size
    gw = W // model.config.patch_size
    L = gh * gw

    grid_thw = x.new_empty((B, 3), dtype=torch.long)
    grid_thw[:, 0] = 1
    grid_thw[:, 1] = gh
    grid_thw[:, 2] = gw

    # patch embedding
    hidden_states = model.patch_embed(x)  # [L, D]

    # position embeddings
    rotary_pos_emb = model.rot_pos_emb(grid_thw)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    position_embeddings = (cos, sin)

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)


    """
        we will run first 3 pre-blocks normally and manually compute attention of 4th block.
    """
    for blk in model.blocks_pre[:3]:
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )

    blk4: Qwen2VLVisionBlock = model.blocks_pre[3]
    normed: torch.Tensor = blk4.norm1(hidden_states)  # [L, D]
    seq_length = normed.shape[0]
    num_heads = blk4.attn.num_heads
    head_dim = blk4.attn.head_dim
    scaling = blk4.attn.scaling
    qkv: torch.Tensor = blk4.attn.qkv(normed)  # [L, 3D]
    query_states, key_states, value_states = (
        qkv.reshape(seq_length, 3, num_heads, head_dim)
        .permute(1, 0, 2, 3)
        .unbind(0)
    ) # q/k/v: [L, H, Hd]

    query_states, key_states = apply_rotary_pos_emb_vision(
        query_states, key_states, cos, sin
    )

    # [L, H, Hd] -> [H, L, Hd]
    q = query_states.permute(1, 0, 2)
    k = key_states.permute(1, 0, 2)
    v = value_states.permute(1, 0, 2)

    # attention
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scaling   # [H, L, L]
    attn_probs = torch.softmax(attn_scores, dim=-1)                # [H, L, L]

    # average over heads, then average over queries
    attn_mean = attn_probs.mean(dim=0)      # [L, L]
    token_score = attn_mean.sum(dim=0)     # [L]

    # min-max normalize for visualization
    attn_map = token_score.view(gh, gw).detach().cpu()
    attn_map = attn_map - attn_map.min()
    attn_map = attn_map / (attn_map.max() + 1e-8)

    return attn_map.numpy(), gh, gw


@torch.no_grad()
def overlay_qwen2vldrip_attention_no_patch(
    model: Qwen2VLDRIP,
    img_3chw: torch.Tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    alpha=0.45,
):
    """
    Returns:
      PIL.Image with layer-4 attention heatmap overlaid on the image.
    """
    attn_map, gh, gw = get_qwen2vldrip_layer4_attention_no_patch(model, img_3chw)

    # unnormalize image for visualization
    orig = unnormalize_img(img_3chw, mean=mean, std=std)
    orig_img = TF.to_pil_image(orig).convert("RGB")
    orig_np = np.array(orig_img).astype(np.float32) / 255.0

    # resize patch attention map to image size
    heatmap_img = Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
        (orig_np.shape[1], orig_np.shape[0]),
        resample=Image.BILINEAR,
    )
    heatmap_np = np.array(heatmap_img).astype(np.float32) / 255.0

    # colorize
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(heatmap_np)[..., :3]

    overlay = (1 - alpha) * orig_np + alpha * heat_rgb
    overlay = np.clip(overlay, 0.0, 1.0)

    return Image.fromarray((overlay * 255).astype(np.uint8))


@torch.no_grad()
def visualize_attention_single_multi_qwen2vldrip(
    model: Qwen2VLDRIP,
    preprocess,
    root_dir="/users/PAS2912/yusenpeng/Fast-CLIP/src/boundary_vis/image_samples/single_multi/",
    save_path="/users/PAS2912/yusenpeng/Fast-CLIP/src/boundary_vis/single_multi_attention_overlay.png",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """
    Reads:
      single_1.JPEG, multi_1.JPEG, single_2.JPEG, multi_2.JPEG
    and plots attention overlays in a separate 2x2 grid.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    names = ["single_1", "multi_1", "single_2", "multi_2"]
    paths = [os.path.join(root_dir, f"{n}.JPEG") for n in names]

    tensors = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")
        tensors.append(load_img_norm(p, preprocess))

    overlays = [
        overlay_qwen2vldrip_attention_no_patch(model, t, mean=mean, std=std)
        for t in tensors
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    plot_titles = ["single_1", "multi_1", "single_2", "multi_2"]

    for ax, ov, title in zip(axes, overlays, plot_titles):
        ax.imshow(ov)
        ax.set_title(f"{title} - attention @ layer 4")
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"Saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def main():
    ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/imagenet_QwenDRIP_4x/model_19.pth"


    patch_size = 16
    COMPRESSION_RATE = 0.25
    config = Qwen2VLVisionConfig(
        depth=12,
        embed_dim=768,
        hidden_size=768 * 4,
        mlp_ratio=4.0,
        num_heads=768 // 64,
        in_channels=3,
        patch_size=patch_size,
        spatial_merge_size=1,
        temporal_patch_size=1,
    )
    model = Qwen2VLDRIP(
        config=config,
        depth=(4, 8, 0),
        temp=0.5,
        compression_rate=COMPRESSION_RATE,
        threshold=0.5
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_qwen2vldrip_checkpoint(
        model,
        checkpoint_path=ckpt_path,
        device=device,
        strict=False,
        verbose=True,
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # converts to [0,1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    visualize_boundaries_single_multi_qwen2vldrip(
        model,
        preprocess,
        root_dir="/users/PAS2912/yusenpeng/Fast-CLIP/src/boundary_vis/image_samples/single_multi/",
        save_path="/users/PAS2912/yusenpeng/Fast-CLIP/src/boundary_vis/single_multi_overlay.png",
    )

    visualize_attention_single_multi_qwen2vldrip(
        model,
        preprocess,
        root_dir="/users/PAS2912/yusenpeng/Fast-CLIP/src/boundary_vis/image_samples/single_multi/",
        save_path="/users/PAS2912/yusenpeng/Fast-CLIP/src/boundary_vis/single_multi_attention_overlay.png"
    )


if __name__ == "__main__":
    main()