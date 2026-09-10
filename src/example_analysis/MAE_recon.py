import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

"""
How to run this script:

salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --partition debug-quad --time 00:30:00
module load miniconda3/24.1.2-py310
conda activate DRIP_flash
python src/example_analysis/MAE_recon.py
"""

PROJECT_ROOT = "/users/PAS2912/yusenpeng/DRIP"
sys.path.insert(0, PROJECT_ROOT)

import src.example_analysis.mae_utils.models_mae as models_mae
from src.boundary_visual_LLaVA import load_img_with_processor, overlay_llava_drip_boundaries, build_llava_drip_vision_tower


MASKING_TYPE = "Fixed"  # "random" or "Fixed" or "DRIP"
MASK_RATIO = 0.75
DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_pretrain_NEW_DOWN_temp001_train_full/drip.bin"


COMPRESSION_RATE = 1 - MASK_RATIO
DEVICE = "cuda"
CHECKPOINT_PATH = "/users/PAS2912/yusenpeng/mae_pretrain_vit_huge_full.pth"
# IMAGE_PATH = "/users/PAS2912/yusenpeng/DRIP/src/example_analysis/TextVQA_results/subset_images/05fab8d9991ca41c.jpg"
IMAGE_PATH = "/users/PAS2912/yusenpeng/DRIP/src/example_analysis/TextVQA_results/subset_images/0c0a22bfd0da315a.jpg"
OUTPUT_PATH = f"/users/PAS2912/yusenpeng/DRIP/src/example_analysis/{MASKING_TYPE.lower()}_mae_reconstruction.png"
VISION_TOWER_NAME = "openai/clip-vit-large-patch14-336"

if MASKING_TYPE == "DRIP":
    vision_model = build_llava_drip_vision_tower(
        vision_tower_name=VISION_TOWER_NAME,
        mm_vision_select_layer=-1,
        mm_vision_select_feature="patch",
        compression_rate=COMPRESSION_RATE,
        drip_weight_path=DRIP_WEIGHT_PATH,
        merge_strategy="DRIP",
        device=DEVICE,
    )
    clip_img_tensor = load_img_with_processor(IMAGE_PATH, vision_model.image_processor)
    with torch.no_grad():
        (
            _,
            drip_hard_mask,
            _,
            drip_num_boundaries,
        ) = overlay_llava_drip_boundaries(
            vision_model,
            clip_img_tensor,
            alpha=0.4,
        )
    print(drip_hard_mask.shape)
    drip_boundary_mask = torch.tensor(drip_hard_mask, dtype=torch.float32, device=DEVICE).flatten().unsqueeze(0)
    print(drip_boundary_mask.shape)
    boundary_mask = drip_boundary_mask
else:
    boundary_mask = None


# load the model
RESOLUTION = 336
model = models_mae.mae_vit_huge_patch14(img_size=RESOLUTION)
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
state_dict = checkpoint["model"]

# The pretrained checkpoint contains 16x16 positional embeddings for 224x224 images; 
# Our model has already initialized fresh 24x24 sine-cosine embeddings for 336x336 images
state_dict.pop("pos_embed", None)
state_dict.pop("decoder_pos_embed", None)
msg = model.load_state_dict(state_dict, strict=False)
model = model.to(DEVICE)
model.eval()

# Load image
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
image = Image.open(IMAGE_PATH).convert("RGB")
transform = transforms.Compose([
    transforms.Resize(
        (RESOLUTION, RESOLUTION),
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN.tolist(),
        std=IMAGENET_STD.tolist(),
    ),
])
x: torch.Tensor = transform(image)
x = x.unsqueeze(0).to(DEVICE)

# MAE reconstruction
with torch.no_grad():
    loss, pred, mask = model(x, mask_ratio=MASK_RATIO, masking_type=MASKING_TYPE, boundary_mask=boundary_mask)
print("Reconstruction loss:", loss.item())
print("Pred:", pred.shape)
print("Mask:", mask.shape)

# Convert predicted patches -> image
pred_img = model.unpatchify(pred)
pred_img = torch.einsum("nchw->nhwc", pred_img)
pred_img = pred_img.detach().cpu().numpy()
original = torch.einsum("nchw->nhwc",x)
original = original.detach().cpu().numpy()
original = (original * IMAGENET_STD + IMAGENET_MEAN)

# Unnormalize MAE prediction
pred_img = (pred_img * IMAGENET_STD + IMAGENET_MEAN)

# Convert patch mask -> pixel mask
# mask: 0 = visible patch and 1 = masked patch
mask_pixels = mask.unsqueeze(-1)
# Each MAE patch is 16 x 16 x 3
mask_pixels = mask_pixels.repeat(
    1,
    1,
    model.patch_embed.patch_size[0] ** 2 * 3,
)
mask_pixels = model.unpatchify(mask_pixels)
mask_pixels = torch.einsum("nchw->nhwc", mask_pixels)
mask_pixels = mask_pixels.detach().cpu().numpy()


# Visible input
masked_image = (original * (1 - mask_pixels))


# Combined reconstruction
# visible regions  -> original pixels
# masked regions   -> MAE prediction
reconstruction = (original * (1 - mask_pixels) + pred_img * mask_pixels)


# Clamp for visualization
original = np.clip(original[0], 0, 1)
masked_image = np.clip(masked_image[0], 0, 1)
pred_img = np.clip(pred_img[0], 0, 1)
reconstruction = np.clip(reconstruction[0], 0, 1)


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(16, 4),
)

axes[0].imshow(original)
axes[0].set_title("Original")

axes[1].imshow(masked_image)
axes[1].set_title(
    f"Masked ({MASK_RATIO:.0%})"
)

axes[2].imshow(pred_img)
axes[2].set_title("MAE Prediction")

axes[3].imshow(reconstruction)
axes[3].set_title(f"Reconstruction with loss {loss.item():.4f}")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(
    f"Saved reconstruction to: {OUTPUT_PATH}"
)
