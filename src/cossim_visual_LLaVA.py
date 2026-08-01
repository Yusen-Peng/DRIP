import os
import sys
from types import SimpleNamespace

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA
PROJECT_ROOT = "/users/PAS2912/yusenpeng/DRIP"
sys.path.insert(0, PROJECT_ROOT)
from src.LLaVA_wrapper.llava_local.model.multimodal_encoder.clip_encoder import CLIPVisionTower


def compute_pca_pc1_heatmap(patch_tokens, grid_h, grid_w, normalize_tokens=True):
    x = patch_tokens.float()
    if normalize_tokens:
        x = torch.nn.functional.normalize(x, dim=-1)
    x_np = x.numpy()  # [L, D]
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(x_np).squeeze(-1)  # [L]
    heat: np.ndarray = minmax_norm(pc1)
    return heat.reshape(grid_h, grid_w), float(pca.explained_variance_ratio_[0])




def unnormalize_img(
    img_3chw,
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
):
    return TF.normalize(
        img_3chw.detach().clone().cpu(),
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1.0 / s for s in std],
    ).clamp(0, 1)


def load_img_with_processor(img_path, image_processor):
    img = Image.open(img_path).convert("RGB")
    out = image_processor(images=img, return_tensors="pt")
    return out["pixel_values"][0]


def build_llava_drip_vision_tower(
    vision_tower_name="openai/clip-vit-large-patch14-336",
    mm_vision_select_layer=-1,
    mm_vision_select_feature="patch",
    compression_rate=0.125,
    drip_weight_path="/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_TEST_DRIP/checkpoint-60/drip.bin",
    device="cuda",
):
    args = SimpleNamespace(
        mm_vision_tower=vision_tower_name,
        mm_vision_select_layer=mm_vision_select_layer,
        mm_vision_select_feature=mm_vision_select_feature,
        unfreeze_mm_vision_tower=False,
    )

    model = CLIPVisionTower(
        vision_tower=vision_tower_name,
        args=args,
        merge_strategy="DRIP",
        compression_rate=compression_rate,
        drip_weight_path=drip_weight_path,
        delay_load=False,
    )
    model = model.to(device).eval()
    return model


@torch.no_grad()
def get_llava_drip_hard_boundaries(model: CLIPVisionTower, img_3chw: torch.Tensor, verbose=False):
    device = model.device
    dtype = model.dtype

    x = img_3chw.unsqueeze(0).to(device=device, dtype=dtype)
    image_forward_outs = model.vision_tower(x, output_hidden_states=True)
    patch_tokens = model.feature_select(image_forward_outs)  # [1, L, D]

    B, L_patch, _ = patch_tokens.shape
    assert B == 1, "Only single-image inference is supported."

    grid_h = model.num_patches_per_side
    grid_w = model.num_patches_per_side
    assert L_patch == grid_h * grid_w, f"Expected {grid_h * grid_w} patches, got {L_patch}"

    patch_transposed = patch_tokens.transpose(0, 1)  # [L, B, D]
    _, hard_boundaries = model.boundary_predictor.inference(patch_transposed, verbose=verbose)

    return hard_boundaries[0].detach().float().cpu(), grid_h, grid_w


@torch.no_grad()
def overlay_llava_drip_boundaries(
    model: CLIPVisionTower,
    img_3chw: torch.Tensor,
    alpha=0.4,
    verbose=False,
):
    hard_1d, grid_h, grid_w = get_llava_drip_hard_boundaries(model, img_3chw, verbose=verbose)
    hard_mask = hard_1d.view(grid_h, grid_w).numpy()

    orig = unnormalize_img(img_3chw)
    orig_img = TF.to_pil_image(orig).convert("RGB")
    orig_np = np.array(orig_img).astype(np.uint8)

    img_h, img_w = orig_np.shape[:2]
    patch_h = img_h // grid_h
    patch_w = img_w // grid_w

    overlay_np = orig_np.copy()
    for i in range(grid_h):
        for j in range(grid_w):
            if hard_mask[i, j] == 1.0:
                y0, y1 = i * patch_h, (i + 1) * patch_h
                x0, x1 = j * patch_w, (j + 1) * patch_w

                patch = overlay_np[y0:y1, x0:x1]
                
                red = np.zeros_like(patch)
                red[..., 0] = 255
                overlay_np[y0:y1, x0:x1] = ((1 - alpha) * patch + alpha * red).astype(np.uint8)
                # color = np.zeros_like(patch)
                # color[..., 1] = 255   # G
                # color[..., 2] = 255   # B  -> cyan

                # overlay_np[y0:y1, x0:x1] = ((1 - alpha) * patch + alpha * color).astype(np.uint8)

    return Image.fromarray(overlay_np), hard_mask


def visualize_10_images_2x5(
    model: CLIPVisionTower,
    image_paths,
    save_path,
    alpha=0.4,
    titles=None,
    verbose=False,
    figsize=(20, 8),
    dpi=300,
    title_fontsize=12,
):
    """
    Process exactly 10 images and save one 2x5 figure.
    """
    assert len(image_paths) == 10, f"Expected exactly 10 images, got {len(image_paths)}"

    overlays = []
    for p in image_paths:
        img_tensor = load_img_with_processor(p, model.image_processor)
        overlay_pil, _ = overlay_llava_drip_boundaries(
            model,
            img_tensor,
            alpha=alpha,
            verbose=verbose,
        )
        overlays.append(overlay_pil)

    if titles is None:
        titles = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

    fig, axes = plt.subplots(2, 5, figsize=figsize)
    axes = axes.flatten()

    for ax, ov, t in zip(axes, overlays, titles):
        ax.imshow(ov)
        ax.set_title(t, fontsize=title_fontsize)
        ax.axis("off")
    plt.tight_layout()
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    print(f"Saved 2x5 figure to: {save_path}")
    plt.close(fig)


def visualize_original_10_images_2x5(
    model: CLIPVisionTower,
    image_paths,
    save_path,
    titles=None,
    figsize=(20, 8),
    dpi=300,
    title_fontsize=12,
):
    """
    Save the original images in the same 2x5 layout/order used for overlays.
    """
    assert len(image_paths) == 10, f"Expected exactly 10 images, got {len(image_paths)}"

    originals = []
    for p in image_paths:
        img_tensor = load_img_with_processor(p, model.image_processor)
        orig = unnormalize_img(img_tensor)
        orig_pil = TF.to_pil_image(orig).convert("RGB")
        originals.append(orig_pil)

    if titles is None:
        titles = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

    fig, axes = plt.subplots(2, 5, figsize=figsize)
    axes = axes.flatten()

    for ax, img, t in zip(axes, originals, titles):
        ax.imshow(img)
        ax.set_title(t, fontsize=title_fontsize)
        ax.axis("off")

    plt.tight_layout()

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    print(f"Saved original 2x5 figure to: {save_path}")
    plt.close(fig)

@torch.no_grad()
def get_llava_patch_tokens(model: CLIPVisionTower, img_3chw: torch.Tensor):
    device = model.device
    dtype = model.dtype

    x = img_3chw.unsqueeze(0).to(device=device, dtype=dtype)
    image_forward_outs = model.vision_tower(x, output_hidden_states=True)
    patch_tokens = model.feature_select(image_forward_outs)  # [1, L, D]

    B, L, D = patch_tokens.shape
    assert B == 1

    grid_h = model.num_patches_per_side
    grid_w = model.num_patches_per_side
    assert L == grid_h * grid_w

    return patch_tokens[0].detach().float().cpu(), grid_h, grid_w

def minmax_norm(x, eps=1e-8):
    x = np.asarray(x)
    return (x - x.min()) / (x.max() - x.min() + eps)


def compute_feature_difference_1D(patch_tokens, grid_h, grid_w):
    x = torch.nn.functional.normalize(patch_tokens, dim=-1)
    grid = x.view(grid_h, grid_w, -1)
    heat = torch.zeros(grid_h, grid_w)
    count = torch.zeros(grid_h, grid_w)

    for i in range(grid_h):
        for j in range(grid_w):
            current = grid[i, j]
            neighbors = []
            if j - 1 >= 0:      # left
                neighbors.append(grid[i, j - 1])
            if j + 1 < grid_w:  # right
                neighbors.append(grid[i, j + 1])
            for neighbor in neighbors:
                diff = 1.0 - torch.dot(current, neighbor)
                heat[i, j] += diff
                count[i, j] += 1
    heat = heat / count.clamp_min(1)
    return minmax_norm(heat.numpy())


def compute_feature_difference_2D(patch_tokens, grid_h, grid_w):
    x = torch.nn.functional.normalize(patch_tokens, dim=-1)
    grid = x.view(grid_h, grid_w, -1)
    heat = torch.zeros(grid_h, grid_w)
    count = torch.zeros(grid_h, grid_w)

    for i in range(grid_h):
        for j in range(grid_w):
            current = grid[i, j]
            neighbors = []
            if i - 1 >= 0:      # up
                neighbors.append(grid[i - 1, j])
            if i + 1 < grid_h:  # down
                neighbors.append(grid[i + 1, j])
            if j - 1 >= 0:      # left
                neighbors.append(grid[i, j - 1])
            if j + 1 < grid_w:  # right
                neighbors.append(grid[i, j + 1])
            for neighbor in neighbors:
                diff = 1.0 - torch.dot(current, neighbor)
                heat[i, j] += diff
                count[i, j] += 1
    heat = heat / count.clamp_min(1)
    return minmax_norm(heat.numpy())

@torch.no_grad()
def visualize_feature_diff_side_by_side_10_images(
    model,
    image_paths,
    save_path,
    mode,
    cmap_name="gray",
    titles=None,
    figsize=(15, 10),  # taller now
    dpi=300,
    title_fontsize=10,
):
    """
    Layout:
    row 0: img1 orig | img2 orig | ... img5
    row 1: img1 heat | img2 heat | ... img5
    row 2: img6 orig | img7 orig | ... img10
    row 3: img6 heat | img7 heat | ... img10
    """
    assert len(image_paths) == 10, f"Expected 10 images, got {len(image_paths)}"

    if titles is None:
        titles = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

    fig, axes = plt.subplots(4, 5, figsize=figsize)

    for idx, p in enumerate(image_paths):
        col = idx % 5

        # top group (images 0–4)
        if idx < 5:
            orig_row = 0
            heat_row = 1
        else:
            orig_row = 2
            heat_row = 3

        img_tensor = load_img_with_processor(p, model.image_processor)

        orig = unnormalize_img(img_tensor)
        orig_pil = TF.to_pil_image(orig).convert("RGB")

        patch_tokens, grid_h, grid_w = get_llava_patch_tokens(model, img_tensor)

        if mode == "1D":
            heat = compute_feature_difference_1D(patch_tokens, grid_h, grid_w)
            heat_title = "1D dissim"
        elif mode == "2D":
            heat = compute_feature_difference_2D(patch_tokens, grid_h, grid_w)
            heat_title = "2D dissim"
        elif mode.upper() == "PCA":
            heat, evr = compute_pca_pc1_heatmap(patch_tokens, grid_h, grid_w)
            heat_title = f"PCA ({evr * 100:.1f}%)"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ---- ORIGINAL ----
        axes[orig_row, col].imshow(orig_pil)
        axes[orig_row, col].set_title(titles[idx], fontsize=title_fontsize)
        axes[orig_row, col].axis("off")

        # ---- HEAT ----
        axes[heat_row, col].imshow(heat, cmap=cmap_name, interpolation="nearest")
        axes[heat_row, col].set_title(heat_title, fontsize=title_fontsize)
        axes[heat_row, col].axis("off")

    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(hspace=0.15, wspace=0.05)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    print(f"Saved {mode} 4x5 layout figure to: {save_path}")
    plt.close(fig)


def compute_adjacent_cosine_sequence(patch_tokens, grid_h, grid_w):
    x = torch.nn.functional.normalize(patch_tokens.float(), dim=-1)  # [L, D]
    L = x.shape[0]
    sims = torch.full((L,), float("nan"), dtype=torch.float32)
    sims[1:] = (x[1:] * x[:-1]).sum(dim=-1)
    heat = sims.view(grid_h, grid_w).cpu().numpy()
    return heat

@torch.no_grad()
def visualize_single_image_sequence_cosine_side_by_side(
    model: CLIPVisionTower,
    image_path,
    save_path,
    cmap_name="bwr",
    figsize=(16, 8),
    dpi=300,
    value_fontsize=5,
    title_fontsize=14,
):
    img_tensor = load_img_with_processor(image_path, model.image_processor)

    orig = unnormalize_img(img_tensor)
    orig_pil = TF.to_pil_image(orig).convert("RGB")

    patch_tokens, grid_h, grid_w = get_llava_patch_tokens(model, img_tensor)
    heat = compute_adjacent_cosine_sequence(patch_tokens, grid_h, grid_w)

    name = os.path.splitext(os.path.basename(image_path))[0]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 1]},
    )

    # ---- ORIGINAL ----
    axes[0].imshow(orig_pil)
    axes[0].set_title(f"{name}: original", fontsize=title_fontsize)
    axes[0].axis("off")

    # ---- HEATMAP ----
    axes[1].imshow(
        heat,
        cmap=cmap_name,
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[1].set_title(
        rf"{name}: cosine $\cos(x_t, x_{{t-1}})$",
        fontsize=title_fontsize,
    )

    # remove EVERYTHING axis-related
    axes[1].axis("off")

    # ---- VALUES ----
    for i in range(grid_h):
        for j in range(grid_w):
            val = heat[i, j]
            text = "NaN" if np.isnan(val) else f"{val:.2f}"

            axes[1].text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=value_fontsize,
                color="black",
            )

    plt.tight_layout()

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    print(f"Saved clean side-by-side figure to: {save_path}")
    plt.close(fig)



def main():
    DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_pretrain/drip.bin"
    COMPRESSION_RATE = 0.25
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_llava_drip_vision_tower(
        vision_tower_name="openai/clip-vit-large-patch14-336",
        mm_vision_select_layer=-1,
        mm_vision_select_feature="patch",
        compression_rate=COMPRESSION_RATE,
        drip_weight_path=DRIP_WEIGHT_PATH,
        device=device,
    )
    image_dir = "/users/PAS2912/yusenpeng/DRIP/src/boundary_vis/LLaVA_examples/"
    image_paths = [
        os.path.join(image_dir, f)
        for f in sorted(os.listdir(image_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]

    for p in image_paths:
        name = os.path.splitext(os.path.basename(p))[0]

        visualize_single_image_sequence_cosine_side_by_side(
            model=model,
            image_path=p,
            save_path=f"/users/PAS2912/yusenpeng/DRIP/src/boundary_vis/LLaVA_results/cosine/{name}_orig_seq_adj_cosine.png",
        )

if __name__ == "__main__":
    main()