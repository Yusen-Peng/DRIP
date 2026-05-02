import os
import sys
from types import SimpleNamespace

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
PROJECT_ROOT = "/users/PAS2912/yusenpeng/DRIP"
sys.path.insert(0, PROJECT_ROOT)
from src.LLaVA_wrapper.llava_local.model.multimodal_encoder.clip_encoder import CLIPVisionTower


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

    """
        enforce the last token to be a boundary token
    """
    last = torch.ones_like(hard_boundaries[:, -1:])
    hard_boundaries = torch.cat([hard_boundaries[:, :-1], last], dim=1)

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


def main():

    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_pretrain/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_finetune_train_lora/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_pretrain_last_force/drip.bin" 
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_pretrain_temp02/checkpoint-530/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_pretrain_temp05/checkpoint-540/drip.bin"
    DRIP_WEIGHT_PATH = None

    COMPRESSION_RATE = 0.25


    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_2x_pretrain/drip.bin"
    # COMPRESSION_RATE = 0.5


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

    assert len(image_paths) == 10, f"Expected exactly 10 images in {image_dir}, found {len(image_paths)}!"

    visualize_original_10_images_2x5(
        model=model,
        image_paths=image_paths,
        save_path="/users/PAS2912/yusenpeng/DRIP/src/boundary_vis/LLaVA_results/llava_originals_2x5.png",
        titles=None,
        figsize=(20, 8),
        dpi=300,
        title_fontsize=12,
    )


    split = DRIP_WEIGHT_PATH.split("/")[5]
    save_path = f"/users/PAS2912/yusenpeng/DRIP/src/boundary_vis/LLaVA_results/{split}.png"
    visualize_10_images_2x5(
        model=model,
        image_paths=image_paths,
        save_path=save_path,
        alpha=0.4,
        titles=None,
        verbose=False,
        figsize=(20, 8),
        dpi=300,
        title_fontsize=12,
    )


if __name__ == "__main__":
    main()