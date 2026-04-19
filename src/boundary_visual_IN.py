import os
from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
from open_clip_local.model import DTPViT

def load_img_norm(img_path, preprocess):
    """
    Returns a normalized tensor [3, H, W].
    Assumes preprocess returns a tensor ready for the model.
    """
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img)
    if isinstance(x, dict):
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


def load_dtpvit_checkpoint(model: DTPViT, checkpoint_path, device="cpu", strict=False, verbose=True):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(ckpt)
    state_dict = _strip_prefix_if_needed(state_dict)

    if verbose:
        print("\n==== BEFORE LOAD ====")
        for name, param in model.boundary_predictor.boundary_predictor.named_parameters():
            d = param.data.float().cpu()
            print(f"{name}: mean={d.mean():.6f}, std={d.std():.6f}")
        
        print("\n==== CHECKPOINT VALUES ====")
        for k, v in state_dict.items():
            if "boundary_predictor.boundary_predictor" in k:
                d = v.float().cpu()
                print(f"{k}: mean={d.mean():.6f}, std={d.std():.6f}")


    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    model.to(device).eval()

    if verbose:
        print(f"[load_dtpvit_checkpoint] loaded from: {checkpoint_path}")
        print(f"  missing keys: {len(missing)}")
        print(f"  unexpected keys: {len(unexpected)}")
        if len(missing) and len(missing) < 50:
            print("  missing:", missing)
        if len(unexpected) and len(unexpected) < 50:
            print("  unexpected:", unexpected)

    return model


@torch.no_grad()
def get_dtpvit_hard_boundaries(model: DTPViT, img_3chw: torch.Tensor):
    """
    For DTPViT:
      input image -> _embeds -> transformer_pre -> boundary_predictor

    Returns:
      hard_boundaries: [L_patch]  (CLS removed)
      grid_h, grid_w
    """
    device = next(model.parameters()).device
    x = img_3chw.unsqueeze(0).to(device)   # [1, 3, H, W]

    B, _, H, W = x.shape
    assert B == 1, "This helper currently assumes a single image."
    gh, gw = model.grid_size
    L_patch = gh * gw

    # embeddings: [B, 1 + L_patch, D]
    hidden_states = model._embeds(x)

    # pre transformer
    hidden_states = model.transformer_pre(hidden_states, attn_mask=None)

    # Split CLS and patch tokens
    cls_token = hidden_states[:, :1, :]      # [B, 1, D]
    patch_tokens = hidden_states[:, 1:, :]   # [B, L, D]

    x_transposed = patch_tokens.transpose(0, 1)  # [1, L, D] -> [L, 1, D]
    
    _, hard_boundaries = model.boundary_predictor.inference(x_transposed, verbose=True)   # [B, L]

    assert hard_boundaries.shape[1] == L_patch, \
        f"Expected {L_patch} patch tokens after removing CLS, got {hard_boundaries.shape[1]}"
    return hard_boundaries[0].detach(), gh, gw


@torch.no_grad()
def overlay_dtpvit_boundaries(
    model: DTPViT,
    img_3chw: torch.Tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """
    Returns:
      PIL.Image with boundary-kept patches overlaid in red.
    """
    hard_1d, grid_h, grid_w = get_dtpvit_hard_boundaries(model, img_3chw)

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
def visualize_boundaries_single_multi_dtpvit(
    model: DTPViT,
    preprocess,
    root_dir="/users/PAS2912/yusenpeng/DRIP/src/boundary_vis/image_samples/single_multi/",
    save_path="/users/PAS2912/yusenpeng/DRIP/src/boundary_vis/single_multi_overlay_dtpvit.png",
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
        overlay_dtpvit_boundaries(model, t, mean=mean, std=std)
        for t in tensors
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    plot_titles = ["single_1", "multi_1", "single_2", "multi_2"]

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
def get_dtpvit_layer4_attention_no_patch(
    model: DTPViT,
    img_3chw: torch.Tensor,
):
    """
    Compute attention map from transformer_pre.resblocks[3] without modifying model code.
    Returns:
        attn_map: [gh, gw] normalized patch heatmap
        gh, gw
    """
    device = next(model.parameters()).device
    x = img_3chw.unsqueeze(0).to(device)  # [1, 3, H, W]

    B, _, H, W = x.shape
    assert B == 1, "This helper currently assumes a single image."
    gh, gw = model.grid_size
    L_patch = gh * gw

    """
        we will run first 3 pre-blocks normally and manually compute attention of 4th block.
    """
    hidden_states = model._embeds(x)   # [B, 1+L, D]

    for blk in model.transformer_pre.resblocks[:3]:
        hidden_states = blk(hidden_states, attn_mask=None)

    blk4 = model.transformer_pre.resblocks[3]
    normed = blk4.ln_1(hidden_states)   # [B, 1+L, D]

    attn = blk4.attn
    B, L_full, C = normed.shape
    num_heads = attn.num_heads
    head_dim = attn.head_dim
    scaling = attn.scale

    """
        Attention.forward():
            if self.batch_first:
                x = x.transpose(0, 1)
            L, N, C = x.shape
            q, k, v = F.linear(...).chunk(3, dim=-1)
            q = q.reshape(L, N * H, Hd).transpose(0, 1)
            ...
    """
    x_attn = normed
    if attn.batch_first:
        x_attn = x_attn.transpose(0, 1)   # [L, B, D]

    seq_length, batch_size, dim = x_attn.shape
    q, k, v = F.linear(x_attn, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)

    q = q.reshape(seq_length, batch_size * num_heads, head_dim).transpose(0, 1)  # [B*H, L, Hd]
    k = k.reshape(seq_length, batch_size * num_heads, head_dim).transpose(0, 1)  # [B*H, L, Hd]
    v = v.reshape(seq_length, batch_size * num_heads, head_dim).transpose(0, 1)  # [B*H, L, Hd]

    if attn.logit_scale is not None:
        attn_scores = torch.bmm(
            F.normalize(q, dim=-1),
            F.normalize(k, dim=-1).transpose(-1, -2)
        )  # [B*H, L, L]
        logit_scale = torch.clamp(attn.logit_scale, max=attn.logit_scale_max).exp()  # [H,1,1]
        attn_scores = attn_scores.view(batch_size, num_heads, seq_length, seq_length) * logit_scale.unsqueeze(0)
        attn_probs = torch.softmax(attn_scores, dim=-1)  # [B, H, L, L]
    else:
        attn_scores = torch.bmm(q * scaling, k.transpose(-1, -2))   # [B*H, L, L]
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = attn_probs.view(batch_size, num_heads, seq_length, seq_length)  # [B, H, L, L]

    # average over heads, then average over queries
    attn_mean = attn_probs[0].mean(dim=0)      # [L, L]
    token_score = attn_mean.sum(dim=0)         # [L]

    # remove CLS token before reshaping
    token_score = token_score[1:]              # [L_patch]

    assert token_score.numel() == L_patch, \
        f"Expected {L_patch} patch tokens after removing CLS, got {token_score.numel()}"

    # min-max normalize for visualization
    attn_map = token_score.view(gh, gw).detach().cpu()
    attn_map = attn_map - attn_map.min()
    attn_map = attn_map / (attn_map.max() + 1e-8)

    return attn_map.numpy(), gh, gw


@torch.no_grad()
def overlay_dtpvit_attention_no_patch(
    model: DTPViT,
    img_3chw: torch.Tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    alpha=0.45,
):
    """
    Returns:
      PIL.Image with layer-4 attention heatmap overlaid on the image.
    """
    attn_map, gh, gw = get_dtpvit_layer4_attention_no_patch(model, img_3chw)

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
def visualize_attention_single_multi_dtpvit(
    model: DTPViT,
    preprocess,
    root_dir="/users/PAS2912/yusenpeng/DRIP/src/boundary_vis/image_samples/single_multi/",
    save_path="/users/PAS2912/yusenpeng/DRIP/src/boundary_vis/single_multi_attention_overlay_dtpvit.png",
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
        overlay_dtpvit_attention_no_patch(model, t, mean=mean, std=std)
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

    ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/imagenet_DRIP_4x_01_warmup2/model_299.pth"

    #ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/imagenet_DRIP_4x_01_warmup5/model_299.pth"
    #ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/imagenet_DRIP_4x_01_warmup5_init/model_299.pth" 


    
    #ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/imagenet_DRIP_4x_half_LR_no_warmup/model_299.pth"
    #ckpt_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/imagenet_DRIP_4x_half_LR_no_warmup_smart_init/model_179.pth"







    patch_size = 16
    COMPRESSION_RATE = 0.25
    RESOLUTION = 224

    width = 768
    model = DTPViT(
        image_size=RESOLUTION,
        patch_size=patch_size,
        width=width,
        layers=12,
        depth=(4, 8, 0),
        compression_rate=COMPRESSION_RATE,
        heads=width // 64,
        mlp_ratio=4.0,
        temp=0.5,
        output_dim=512,
        pos_embed_type='sin_cos_2d', # 'learnable' or 'sin_cos_2d'
        pool_type='avg'
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_dtpvit_checkpoint(
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


    visualize_boundaries_single_multi_dtpvit(
        model,
        preprocess,
        root_dir="/users/PAS2912/yusenpeng/DRIP/src/boundary_vis/image_samples/single_multi/",
        save_path="/users/PAS2912/yusenpeng/DRIP/src/boundary_vis/dtpvit_single_multi_overlay.png",
    )
    visualize_attention_single_multi_dtpvit(
        model,
        preprocess,
        root_dir="/users/PAS2912/yusenpeng/DRIP/src/boundary_vis/image_samples/single_multi/",
        save_path="/users/PAS2912/yusenpeng/DRIP/src/boundary_vis/dtpvit_single_multi_attention_overlay.png"
    )

if __name__ == "__main__":
    main()