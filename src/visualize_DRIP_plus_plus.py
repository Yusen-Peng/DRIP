import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from LLaVA_wrapper.llava_local.model.builder import load_pretrained_model
from LLaVA_wrapper.llava_local.mm_utils import process_images, get_model_name_from_path
from LLaVA_wrapper.llava_local.utils import disable_torch_init

"""
salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --partition debug-nextgen --time 00:30:00
module load miniconda3/24.1.2-py310
conda activate DRIP_flash

python src/visualize_DRIP_plus_plus.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_pretrain_NEW_DOWN_temp001_train_full \
    --image-path /users/PAS2912/yusenpeng/DRIP/src/CDPruner_imgs/pancake.jpg \
    --alpha 0.25 \
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
    model.eval()
    vision_tower = model.get_vision_tower()

    # Ensure all DRIP++ components are on the same device.
    device = next(vision_tower.vision_tower.parameters()).device
    dtype = next(vision_tower.vision_tower.parameters()).dtype

    vision_tower.boundary_predictor.to(device=device, dtype=dtype)
    vision_tower.null_token.data = vision_tower.null_token.data.to(device=device, dtype=dtype)

    assert vision_tower.merge_strategy == "DRIP++", "This script requires merge_strategy=DRIP++"
    return model, image_processor


def load_image(args, image_processor, model):
    image = Image.open(args.image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)

    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor)

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    vision_tower = model.get_vision_tower()

    vision_dtype = next(vision_tower.vision_tower.parameters()).dtype
    image_tensor = image_tensor.to(device=vision_tower.device, dtype=vision_dtype)
    return image_tensor


def recover_display_image(image_tensor, image_processor):
    x = image_tensor[0].float().cpu()

    mean = torch.tensor(
        image_processor.image_mean,
        dtype=x.dtype,
    ).view(3, 1, 1)

    std = torch.tensor(
        image_processor.image_std,
        dtype=x.dtype,
    ).view(3, 1, 1)

    x = x * std + mean
    x = x.clamp(0, 1)

    return x.permute(1, 2, 0).numpy()


def select_boundaries(scores, compression_rate):
    """Reproduce DRIP++ TopK boundary selection."""
    B, N = scores.shape
    k = max(1, min(round(N * compression_rate), N))
    boundaries = torch.zeros_like(scores)
    if k > 1:
        indices = torch.topk(scores[:, :-1], k=k - 1, dim=-1).indices
        boundaries.scatter_(1, indices, 1.0)
    boundaries[:, -1] = 1.0
    return boundaries.bool()


@torch.inference_mode()
def run_drip_plus_plus(model, image_tensor, prompts, alpha):
    vision_tower = model.get_vision_tower()

    grid_size = vision_tower.num_patches_per_side
    H = W = grid_size

    # Extract the exact CLIP features used by DRIP++.
    image_forward_outs = vision_tower.vision_tower(
        image_tensor,
        output_hidden_states=True,
    )

    image_features = image_forward_outs.hidden_states[
        vision_tower.select_layer
    ]

    # Remove CLS token if patch features are used.
    if vision_tower.select_feature == "patch":
        image_features = image_features[:, 1:]

    # Original DRIP boundary predictor.
    boundary_scores, _ = vision_tower.boundary_predictor.inference(image_features.transpose(0, 1))

    if boundary_scores.ndim == 3:
        assert boundary_scores.shape[-1] == 1
        boundary_scores = boundary_scores.squeeze(-1)

    boundary_scores = boundary_scores.float()

    results = []

    print("\n" + "=" * 70)
    print("DRIP++ Boundary and Relevance Analysis")
    print("=" * 70)

    for i, prompt in enumerate(prompts):
        print(f"\n[{i + 1}/{len(prompts)}] Prompt: {prompt}")

        # Use the actual DRIP++ relevance implementation.
        relevance = vision_tower.compute_prompt_relevance(image_features, [prompt])

        if relevance.ndim == 3:
            assert relevance.shape[-1] == 1
            relevance = relevance.squeeze(-1)

        relevance = relevance.float()

        adjusted_scores = alpha * relevance + (1 - alpha) * boundary_scores
        original_mask = select_boundaries(boundary_scores, vision_tower.compression_rate)
        adjusted_mask = select_boundaries(adjusted_scores, vision_tower.compression_rate)
        overlap = (original_mask & adjusted_mask).sum().item()
        num_selected = original_mask.sum().item()
        overlap_ratio = overlap / num_selected

        print(f"  Boundary overlap: {overlap_ratio:.4f}")

        results.append({
            "prompt": prompt,
            "boundary": boundary_scores[0]
                .reshape(H, W)
                .cpu()
                .numpy(),

            "relevance": relevance[0]
                .reshape(H, W)
                .cpu()
                .numpy(),

            "adjusted": adjusted_scores[0]
                .reshape(H, W)
                .cpu()
                .numpy(),

            "original_mask": original_mask[0]
                .reshape(H, W)
                .cpu()
                .numpy(),

            "adjusted_mask": adjusted_mask[0]
                .reshape(H, W)
                .cpu()
                .numpy(),

            "overlap": overlap_ratio,
        })

    return results


def visualize_maps(display_image, results, alpha, output_dir):
    num_prompts = len(results)

    fig, axes = plt.subplots(
        num_prompts,
        3,
        figsize=(15, 5 * num_prompts),
        squeeze=False,
    )

    titles = [
        "Original DRIP Boundary",
        "CLIP Prompt Relevance",
        f"Adjusted Boundary (alpha={alpha})",
    ]

    image_h, image_w = display_image.shape[:2]

    for row, result in enumerate(results):
        maps = [
            result["boundary"],
            result["relevance"],
            result["adjusted"],
        ]

        masks = [
            result["original_mask"],
            None,
            result["adjusted_mask"],
        ]

        for col, (score_map, mask) in enumerate(zip(maps, masks)):
            ax = axes[row, col]

            ax.imshow(display_image)

            heatmap = ax.imshow(
                score_map,
                cmap="jet",
                alpha=0.65,
                interpolation="nearest",
                vmin=0,
                vmax=1,
                extent=(0, image_w, image_h, 0),
            )

            # Show selected boundaries as white dots.
            if mask is not None:
                ys, xs = np.where(mask)

                patch_h = image_h / mask.shape[0]
                patch_w = image_w / mask.shape[1]

                ax.scatter(
                    (xs + 0.5) * patch_w,
                    (ys + 0.5) * patch_h,
                    s=11,
                    c="white",
                    marker="o",
                    edgecolors="black",
                    linewidths=0.2,
                )

            ax.set_title(
                result["prompt"] if col == 1 else titles[col],
                fontsize=15,
                fontweight="bold",
            )

            ax.axis("off")

        axes[row, 0].set_ylabel(
            result["prompt"],
            fontsize=12,
        )

        axes[row, 2].text(
            0.5,
            -0.04,
            f"Boundary overlap: {result['overlap']:.1%}",
            transform=axes[row, 2].transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )

    plt.tight_layout()

    output_path = os.path.join(output_dir, "drip_plus_plus_analysis.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

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

    results = run_drip_plus_plus(
        model,
        image_tensor,
        args.prompts,
        args.alpha,
    )

    visualize_maps(
        display_image,
        results,
        args.alpha,
        args.output_dir,
    )

    print("\n[DONE] DRIP++ visualization completed!")


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
        default="/users/PAS2912/yusenpeng/DRIP/src/drip_plus_plus_visualizations",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--prompts",
        nargs="+",
        required=True,
    )

    args = parser.parse_args()

    assert 0 <= args.alpha <= 1

    main(args)