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
from src.boundary_visual_LLaVA import load_img_with_processor, overlay_llava_drip_boundaries, build_llava_drip_vision_tower, overlay_llava_fixed_boundaries

from src.LLaVA_wrapper.llava_local.model.multimodal_encoder.siglip_encoder import SiglipVisionTower
from src.boundary_visual_LLaVA import build_llava_siglip_vision_tower



def visualize_single_image(
    model: CLIPVisionTower,
    drip: bool,
    image_path,
    save_path: str,
    alpha=0.4,
    titles=None,
    verbose=False,
    figsize=(10,10),
    dpi=300,
    title_fontsize=12,
):

    overlays = []
    num_boundary_patches_list = []
    soft_masks = []


    img_tensor = load_img_with_processor(image_path, model.image_processor)
    if drip:
        overlay_pil, _, soft_mask, num_boundary_patches = overlay_llava_drip_boundaries(
            model,
            img_tensor,
            alpha=alpha,
            verbose=verbose,
        )
    else:
        overlay_pil, _, soft_mask, num_boundary_patches = overlay_llava_fixed_boundaries(
            model,
            img_tensor,
            alpha=alpha,
            verbose=verbose,
        )
    overlays.append(overlay_pil)
    num_boundary_patches_list.append(int(num_boundary_patches))
    soft_masks.append(soft_mask)

    # create a single figure
    fig, axes = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    axes.imshow(overlays[0])
    axes.axis('off')
    if titles is not None:
        axes.set_title(f"{titles[0]}\nNum Boundary Patches: {num_boundary_patches_list[0]}", fontsize=title_fontsize)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    if verbose:
        print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    # IMAGE_ID = "082f5ae635a90152"

    IMAGE_ID = "0a9e38ebdc5bbbf4"
    IMAGE_ID = "00ad17e05c1a1bf5"
    IMAGE_ID = "00e9ff4c6baa2ab6"
    IMAGE_ID = "0ef6658f6d52d8f9"


    MERGE_STRATEGY = "DRIP" # "DRIP" or "DRIP-H"
    COMPRESSION_RATE = 0.25

    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_finetune_train_lora/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_8x_finetune_train_lora/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_10x_finetune_train_lora/drip.bin"

    
    

    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_4x_train_full/drip.bin"
    DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_4x_temp01_new_downsample_train_full/drip.bin"




    device = "cuda" if torch.cuda.is_available() else "cpu"


    if "clip" in DRIP_WEIGHT_PATH.lower():
        model = build_llava_drip_vision_tower(
            vision_tower_name="openai/clip-vit-large-patch14-336",
            mm_vision_select_layer=-1,
            mm_vision_select_feature="patch",
            compression_rate=COMPRESSION_RATE,
            drip_weight_path=DRIP_WEIGHT_PATH,
            merge_strategy=MERGE_STRATEGY,
            device=device,
        )
    elif "siglip" in DRIP_WEIGHT_PATH.lower():
        model = build_llava_siglip_vision_tower(
            vision_tower_name="google/siglip2-large-patch16-384",
            mm_vision_select_layer=-1,
            mm_vision_select_feature="patch",
            compression_rate=COMPRESSION_RATE,
            drip_weight_path=DRIP_WEIGHT_PATH,
            merge_strategy=MERGE_STRATEGY,
            device=device,
        )
    else:
        raise ValueError(f"Unknown DRIP_WEIGHT_PATH: {DRIP_WEIGHT_PATH}")


    ################ ORIGINAL CLIP analysis ################
    # image_path = f"/users/PAS2912/yusenpeng/DRIP/src/example_analysis/TextVQA_results/original_images/{IMAGE_ID}.jpg"
    # save_path = f"/users/PAS2912/yusenpeng/DRIP/src/example_analysis/TextVQA_results/boundary_maps/{IMAGE_ID}_DRIP_overlay.png"
    # visualize_single_image(model, drip=True, image_path=image_path, save_path=save_path, alpha=0.4, verbose=True)
    # save_path = f"/users/PAS2912/yusenpeng/DRIP/src/example_analysis/TextVQA_results/boundary_maps/{IMAGE_ID}_FIXED_overlay.png"
    # visualize_single_image(model, drip=False, image_path=image_path, save_path=save_path, alpha=0.4, verbose=True)


    ################ Teaser Image generation ################
    # image_path = f"/users/PAS2912/yusenpeng/DRIP/src/example_analysis/MME_results/original_images/MME_example.jpg"
    # save_path = f"/users/PAS2912/yusenpeng/DRIP/src/example_analysis/MME_results/boundary_maps/MME_example_DRIP_overlay.png"
    # visualize_single_image(model, drip=True, image_path=image_path, save_path=save_path, alpha=0.4, verbose=True)
    # save_path = f"/users/PAS2912/yusenpeng/DRIP/src/example_analysis/MME_results/boundary_maps/MME_example_FIXED_overlay.png"
    # visualize_single_image(model, drip=False, image_path=image_path, save_path=save_path, alpha=0.4, verbose=True)



    image_path = f"/users/PAS2912/yusenpeng/DRIP/src/example_analysis/TextVQA_results/siglip_diagnose_cases/{IMAGE_ID}.jpg"
    if "new_downsample" in DRIP_WEIGHT_PATH.lower():
        save_path = f"/users/PAS2912/yusenpeng/DRIP/src/example_analysis/TextVQA_results/siglip_boundary_maps/{IMAGE_ID}_NEW_overlay.png"
    else:
        save_path = f"/users/PAS2912/yusenpeng/DRIP/src/example_analysis/TextVQA_results/siglip_boundary_maps/{IMAGE_ID}_OLD_overlay.png"
    visualize_single_image(model, drip=True, image_path=image_path, save_path=save_path, alpha=0.4, verbose=True)


