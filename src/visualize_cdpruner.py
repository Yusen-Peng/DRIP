import argparse
import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from LLaVA_wrapper.llava_local.model.builder import load_pretrained_model
from LLaVA_wrapper.llava_local.mm_utils import process_images, get_model_name_from_path
from LLaVA_wrapper.llava_local.utils import disable_torch_init
from LLaVA_wrapper.llava_local.model.llava_arch import encode_images_cdpruner



"""
How to run this script:

salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --partition debug-nextgen --time 00:30:00
module load miniconda3/24.1.2-py310
conda activate DRIP_flash
python src/visualize_cdpruner.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_FLASH_finetune_ALL_ONCE_full \
    --image-path /users/PAS2912/yusenpeng/DRIP/src/CDPruner_imgs/pancake.jpg \
    --prompts \
        "How many pancakes are in the image?" \
        "How many clusters of grapes are in the image?"
"""


def load_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"\n[INFO] Loading model: {model_path}")
    _, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    model = model.float()
    model.eval()
    vision_tower = model.get_vision_tower()
    projector = model.get_model().mm_projector
    projector_dtype = next(projector.parameters()).dtype
    print(f"[INFO] Projector dtype: {projector_dtype}")
    # CLIP vision model can output float32 even when the projector is float16.
    vision_tower.vision_tower.to(dtype=projector_dtype)

    print(f"[INFO] Vision tower: {vision_tower.vision_tower_name}")
    print(f"[INFO] Merge strategy: {vision_tower.merge_strategy}")
    print(f"[INFO] Compression rate: {vision_tower.compression_rate}")
    assert vision_tower.merge_strategy == "CDPruner", "This script requires merge_strategy='CDPruner'."
    return model, image_processor

def load_image(args, image_processor, model):
    image = Image.open(args.image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor)
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    vision_tower = model.get_vision_tower()
    image_tensor = image_tensor.to(device=vision_tower.device, dtype=torch.float32)
    print(f"[INFO] Original image size: {image.size}")
    print(f"[INFO] Image tensor shape: {image_tensor.shape}")
    return image_tensor

def recover_display_image(image_tensor, image_processor):
    x = image_tensor[0].float().cpu()
    mean = torch.tensor(image_processor.image_mean, dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor(image_processor.image_std, dtype=x.dtype).view(3, 1, 1)
    x = x * std + mean
    x = x.clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


@torch.inference_mode()
def run_cdpruner(model, image_tensor, prompts):
    """Run the actual CDPruner selection algorithm."""
    vision_tower = model.get_vision_tower()
    grid_size = vision_tower.num_patches_per_side
    H = W = grid_size
    masks = []
    selected_indices = []
    print("\n" + "=" * 70)
    print("CDPruner Token Selection")
    print("=" * 70)
    for i, prompt in enumerate(prompts):
        print(f"\n[{i + 1}/{len(prompts)}] Prompt: {prompt}")
        # Call the actual implementation from llava_arch.py.
        _, index_masks = encode_images_cdpruner(model, image_tensor, texts=prompt)

        assert index_masks.shape[0] == 1, "Expected batch size 1."
        mask = index_masks[0].detach().cpu().bool()
        assert mask.numel() == H * W, f"Expected {H * W} patch tokens, but got {mask.numel()}."

        indices = torch.where(mask)[0].numpy()
        mask_2d = mask.reshape(H, W).numpy()
        masks.append(mask_2d)
        selected_indices.append(indices)
        print(f"  Total patches: {H * W}")
        print(f"  Selected patches: {len(indices)}")
        print(f"  Retention ratio: {len(indices)/(H*W):.4f}")
    masks = np.stack(masks, axis=0)
    return masks, selected_indices


def visualize_masks(display_image, masks, prompts, output_dir):
    num_prompts = len(prompts)
    fig, axes = plt.subplots(
        1,
        num_prompts + 1,
        figsize=(5 * (num_prompts + 1), 5),
        squeeze=False,
    )
    axes = axes[0]
    axes[0].imshow(display_image)
    axes[0].set_title(
        "Original Image",
        fontsize=18,
        fontweight="bold",
    )
    axes[0].axis("off")
    image_h, image_w = display_image.shape[:2]
    for i, (mask, prompt) in enumerate(zip(masks, prompts)):
        ax = axes[i + 1]
        ax.imshow(display_image)
        # RGBA overlay at patch-grid resolution.
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        # Darken discarded patches.
        overlay[..., :3] = 0.0
        overlay[..., 3] = (~mask).astype(np.float32) * 0.80
        ax.imshow(overlay, interpolation="nearest", extent=(0, image_w, image_h, 0))
        ax.set_title(f"{prompt}", fontsize=13, fontweight="bold")

        ax.axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        "selected_patches.png",
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"[SAVED] {output_path}")



def main(args):

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )
    model, image_processor = load_model(args)
    image_tensor = load_image(
        args,
        image_processor,
        model,
    )

    display_image = recover_display_image(
        image_tensor,
        image_processor,
    )

    masks, selected_indices = run_cdpruner(
        model,
        image_tensor,
        args.prompts,
    )

    visualize_masks(
        display_image,
        masks,
        args.prompts,
        args.output_dir,
    )

    print("\n[DONE] CDPruner visualization completed!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/users/PAS2912/yusenpeng/DRIP/src/CDPruner_imgs",
    )

    parser.add_argument(
        "--prompts",
        nargs="+",
        required=True,
        help="Multiple text prompts for CDPruner.",
    )

    args = parser.parse_args()

    main(args)