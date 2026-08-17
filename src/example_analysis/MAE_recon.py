import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

"""
How to run this script:

salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --partition debug-nextgen --time 00:30:00
module load miniconda3/24.1.2-py310
conda activate DRIP_flash
python src/example_analysis/MAE_recon.py
"""

PROJECT_ROOT = "/users/PAS2912/yusenpeng/DRIP"
sys.path.insert(0, PROJECT_ROOT)

import src.example_analysis.mae_utils.models_mae as models_mae
DEVICE = "cuda"

CHECKPOINT_PATH = "/users/PAS2912/yusenpeng/mae_pretrain_vit_large_full.pth"

IMAGE_PATH = "/users/PAS2912/yusenpeng/DRIP/src/example_analysis/stop_sign.png"

OUTPUT_PATH = "/users/PAS2912/yusenpeng/DRIP/src/example_analysis/mae_reconstruction.png"


# load the model
model = models_mae.mae_vit_large_patch16()
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(checkpoint["model"], strict=True)
model = model.to(DEVICE)
model.eval()

# Load image
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
image = Image.open(IMAGE_PATH).convert("RGB")
transform = transforms.Compose([
    transforms.Resize(
        (224, 224),
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
MASK_RATIO = 0.75
with torch.no_grad():
    loss, pred, mask = model(x, mask_ratio=MASK_RATIO)
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
