import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from transformers import AutoProcessor


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../"))
sys.path.insert(0, PROJECT_ROOT)
from model.qwen3vl_compressed import (
    CompressedQwen3VLForConditionalGeneration,
)


def get_qwen_processed_image(
    original_img,
    model,
    image_grid_thw,
):
    """
    Resize the original image to the spatial resolution represented
    by Qwen's pre-spatial-merge vision grid.
    """
    grid = image_grid_thw[0]

    h = int(grid[1].item())
    w = int(grid[2].item())

    patch_size = int(
        model.config.vision_config.patch_size
    )

    processed_h = h * patch_size
    processed_w = w * patch_size

    processed_img = original_img.resize(
        (processed_w, processed_h),
        resample=Image.BICUBIC,
    )

    return processed_img


def build_qwen3vl_drip(
    model_base="Qwen/Qwen3-VL-4B-Instruct",
    drip_weight_path=None,
    merge_strategy="DRIP",
    compression_rate=0.25,
    temperature=0.1,
    mlp_ratio=1.0,
    device="cuda",
):
    processor = AutoProcessor.from_pretrained(model_base)
    model = CompressedQwen3VLForConditionalGeneration.from_pretrained(
        model_base,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.model.set_compressor(
        merge_strategy=merge_strategy,
        compression_rate=compression_rate,
        temperature=temperature,
        drip_path=drip_weight_path,
        mlp_ratio=mlp_ratio,
    )
    model = model.to(device).eval()
    return model, processor


def prepare_qwen_image(
    img_path,
    processor,
    model,
):
    img = Image.open(img_path).convert("RGB")
    # IMPORTANT:
    # Use the image processor directly.
    # The full Qwen3VL processor expects text + image placeholders.
    inputs = processor.image_processor(
        images=img,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    pixel_values = inputs["pixel_values"].to(
        device=device,
        dtype=dtype,
    )

    image_grid_thw = inputs["image_grid_thw"].to(device)

    return img, pixel_values, image_grid_thw


def get_qwen_llm_grid(
    model,
    image_grid_thw,
    num_visual_tokens,
):
    """
    image_grid_thw is the pre-spatial-merge vision grid.
    Compressor acts on post-merge visual tokens.
    """

    grid = image_grid_thw[0]
    t = int(grid[0].item())
    h = int(grid[1].item())
    w = int(grid[2].item())

    merge_size = int(
        model.config.vision_config.spatial_merge_size
    )

    assert t == 1, (
        f"Expected a single-image temporal dimension of 1, got {t}"
    )

    assert h % merge_size == 0
    assert w % merge_size == 0

    grid_h = h // merge_size
    grid_w = w // merge_size

    expected_tokens = t * grid_h * grid_w

    assert num_visual_tokens == expected_tokens, (
        f"🤬 Qwen grid mismatch:\n"
        f"image_grid_thw={image_grid_thw.tolist()}\n"
        f"spatial_merge_size={merge_size}\n"
        f"post-merge grid={grid_h}x{grid_w}\n"
        f"expected tokens={expected_tokens}\n"
        f"actual image embeds={num_visual_tokens}"
    )

    return grid_h, grid_w


@torch.no_grad()
def get_qwen_drip_boundaries(
    model: CompressedQwen3VLForConditionalGeneration,
    pixel_values,
    image_grid_thw,
):
    qwen_model = model.model
    compressor = qwen_model.visual.merger.compressor

    (
        _,
        _,
        boundaries,
        _,
    ) = qwen_model.get_image_features(
        pixel_values,
        image_grid_thw,
    )

    # boundaries are defined over the ORIGINAL post-2x2-merge
    # visual-token sequence [1, N]
    hard = boundaries[0].detach().float().cpu()


    # Soft probabilities cached during the same inference pass
    if compressor.last_soft_boundaries is None:
        raise RuntimeError("last_soft_boundaries was not populated. Make sure the model is in eval mode and inference=True.")
    soft = compressor.last_soft_boundaries[0].detach().float().cpu()

    num_visual_tokens = hard.numel()
    grid_h, grid_w = get_qwen_llm_grid(
        model,
        image_grid_thw,
        num_visual_tokens=num_visual_tokens,
    )
    assert hard.numel() == grid_h * grid_w
    return (
        hard,
        soft,
        grid_h,
        grid_w,
    )

@torch.no_grad()
def overlay_qwen_drip_boundaries(
    model,
    processor,
    img_path,
    alpha=0.4,
):
    original_img, pixel_values, image_grid_thw = (
        prepare_qwen_image(
            img_path,
            processor,
            model,
        )
    )

    hard_1d, soft_1d, grid_h, grid_w = (
        get_qwen_drip_boundaries(
            model,
            pixel_values,
            image_grid_thw,
        )
    )

    hard_mask = hard_1d.view(
        grid_h,
        grid_w,
    ).numpy()

    soft_mask = soft_1d.view(
        grid_h,
        grid_w,
    ).numpy()

    processed_img = get_qwen_processed_image(
        original_img,
        model,
        image_grid_thw,
    )

    orig_np = np.asarray(processed_img).copy()

    img_h, img_w = orig_np.shape[:2]

    assert img_h % grid_h == 0
    assert img_w % grid_w == 0

    cell_h = img_h // grid_h
    cell_w = img_w // grid_w

    overlay_np = orig_np.copy()

    for i in range(grid_h):
        for j in range(grid_w):

            if hard_mask[i, j] > 0.5:
                y0 = i * cell_h
                y1 = (i + 1) * cell_h

                x0 = j * cell_w
                x1 = (j + 1) * cell_w

                patch = overlay_np[
                    y0:y1,
                    x0:x1,
                ]

                red = np.zeros_like(patch)
                red[..., 0] = 255

                overlay_np[
                    y0:y1,
                    x0:x1,
                ] = (
                    (1 - alpha) * patch
                    + alpha * red
                ).astype(np.uint8)

    num_boundaries = int(
        (hard_1d > 0.5).sum().item()
    )

    return (
        Image.fromarray(overlay_np),
        hard_mask,
        soft_mask,
        num_boundaries,
        grid_h,
        grid_w,
        image_grid_thw.detach().cpu(),
    )



def visualize_qwen_10_images_2x5(
    model,
    processor,
    image_paths,
    save_path,
    alpha=0.4,
    titles=None,
    figsize=(20, 8),
    dpi=300,
    title_fontsize=12,
):
    assert len(image_paths) == 10

    overlays = []
    soft_masks = []
    stats = []

    for p in image_paths:

        (
            overlay,
            hard_mask,
            soft_mask,
            num_boundaries,
            grid_h,
            grid_w,
            image_grid_thw,
        ) = overlay_qwen_drip_boundaries(
            model,
            processor,
            p,
            alpha=alpha,
        )

        total_tokens = grid_h * grid_w

        overlays.append(overlay)
        soft_masks.append(soft_mask)

        stats.append(
            (
                num_boundaries,
                total_tokens,
                grid_h,
                grid_w,
            )
        )

        print(
            f"{os.path.basename(p)}: "
            f"grid={grid_h}x{grid_w}, "
            f"L={total_tokens}, "
            f"boundaries={num_boundaries}, "
            f"rate={num_boundaries / total_tokens:.3f}, "
            # f"soft_mean={soft_mask.mean():.4f}, "
            # f"soft_std={soft_mask.std():.4f}, "
            # f"soft_min={soft_mask.min():.4f}, "
            # f"soft_max={soft_mask.max():.4f}, "
            f"image_grid_thw={image_grid_thw.tolist()}"
        )

    if titles is None:
        titles = [
            os.path.splitext(
                os.path.basename(p)
            )[0]
            for p in image_paths
        ]

    # ============================================================
    # Hard-boundary overlay
    # ============================================================
    fig, axes = plt.subplots(
        2,
        5,
        figsize=figsize,
    )

    axes = axes.flatten()

    for (
        ax,
        overlay,
        title,
        stat,
    ) in zip(
        axes,
        overlays,
        titles,
        stats,
    ):
        (
            num_boundaries,
            total_tokens,
            grid_h,
            grid_w,
        ) = stat

        ax.imshow(overlay)

        ax.set_title(
            f"{title}\n"
            f"{num_boundaries}/{total_tokens} "
            f"({100 * num_boundaries / total_tokens:.1f}%), "
            f"{grid_h}×{grid_w}",
            fontsize=title_fontsize,
        )

        ax.axis("off")

    plt.tight_layout()

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(
        save_path,
        bbox_inches="tight",
        dpi=dpi,
    )

    plt.close(fig)

    print(f"Saved hard-boundary visualization to: {save_path}")

    # ============================================================
    # Soft-boundary probability maps
    # ============================================================
    root, ext = os.path.splitext(save_path)

    soft_save_path = root + "_soft_probs.pdf"

    visualize_qwen_soft_probs(
        soft_masks=soft_masks,
        titles=titles,
        save_path=soft_save_path,
        figsize=figsize,
        dpi=dpi,
    )

    print(f"Saved soft-boundary visualization to: {soft_save_path}")


def visualize_qwen_soft_probs(
    soft_masks,
    titles,
    save_path,
    figsize=(20, 8),
    dpi=300,
):
    fig, axes = plt.subplots(
        2,
        5,
        figsize=figsize,
    )

    axes = axes.flatten()

    for ax, soft_mask, title in zip(
        axes,
        soft_masks,
        titles,
    ):
        soft_mask = np.asarray(soft_mask)

        im = ax.imshow(
            soft_mask,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )

        grid_h, grid_w = soft_mask.shape

        # # I might actually NOT print every value for large
        # # native-resolution grids, because it gets unreadable.
        # if grid_h * grid_w <= 200:
        #     for i in range(grid_h):
        #         for j in range(grid_w):
        #             value = soft_mask[i, j]

        #             ax.text(
        #                 j,
        #                 i,
        #                 f"{value:.2f}",
        #                 ha="center",
        #                 va="center",
        #                 fontsize=3,
        #                 color=(
        #                     "white"
        #                     if value < 0.5
        #                     else "black"
        #                 ),
        #             )

        ax.set_title(
            f"{title}: {grid_h}×{grid_w}",
            fontsize=10,
        )

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    plt.savefig(
        save_path,
        bbox_inches="tight",
        dpi=dpi,
    )

    plt.close(fig)



def main():
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_NEW_PIPELINE_1xwidth/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_DRIP_4x_NEW_PIPELINE_1xwidth_BP_WARMUP/checkpoint-24/drip.bin"


    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_NEW_PIPELINE_2xwidth/drip.bin"

    DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_NEW_PIPELINE_1xwidth_temp10/drip.bin"


    COMPRESSION_RATE = 0.25
    TEMPERATURE = 1.0
    MLP_RATIO = 1.0
    # save_path = "/users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune/qwenvl/boundaries/Qwen3VL_results/Qwen3VL_DRIP_4x_NEW_PIPELINE_1xwidth_BP_WARMUP/checkpoint-24.png"
    save_path = "/users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune/qwenvl/boundaries/Qwen3VL_results/Qwen3VL_SFT_DRIP_4x_NEW_PIPELINE_1xwidth_temp10.png"


    MODEL_BASE = "Qwen/Qwen3-VL-4B-Instruct"
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    model, processor = build_qwen3vl_drip(
        model_base=MODEL_BASE,
        drip_weight_path=DRIP_WEIGHT_PATH,
        merge_strategy="DRIP",
        compression_rate=COMPRESSION_RATE,
        temperature=TEMPERATURE,
        mlp_ratio=MLP_RATIO,
        device=device,
    )
    image_dir = "/users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune/qwenvl/boundaries/image_examples"
    image_paths = [
        os.path.join(image_dir, f)
        for f in sorted(os.listdir(image_dir))
        if f.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            )
        )
    ]

    assert len(image_paths) == 10

    visualize_qwen_10_images_2x5(
        model=model,
        processor=processor,
        image_paths=image_paths,
        save_path=save_path,
        alpha=0.4,
    )

if __name__ == "__main__":
    main()