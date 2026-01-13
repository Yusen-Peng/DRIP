import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def patch_entropy_gray(patch_rgb: np.ndarray, bins: int = 256) -> float:
    """
    Shannon entropy on grayscale histogram.
    patch_rgb: (ph, pw, 3) uint8
    """
    # RGB -> grayscale (luma)
    gray = (
        0.2989 * patch_rgb[..., 0] +
        0.5870 * patch_rgb[..., 1] +
        0.1140 * patch_rgb[..., 2]
    ).astype(np.uint8)

    hist = np.bincount(gray.reshape(-1), minlength=bins).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def entropy_boundary_mask(
    img_rgb: np.ndarray,
    grid_h: int,
    grid_w: int,
    top_frac: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      ent_map: (grid_h, grid_w) entropy values
      hard_mask: (grid_h, grid_w) {0,1} boundary tokens = top_frac entropy
    """
    H, W, _ = img_rgb.shape
    patch_h = H // grid_h
    patch_w = W // grid_w

    # If not divisible, crop to a clean grid
    Hc = patch_h * grid_h
    Wc = patch_w * grid_w
    img_rgb = img_rgb[:Hc, :Wc]
    H, W, _ = img_rgb.shape

    ent = np.zeros((grid_h, grid_w), dtype=np.float64)

    for i in range(grid_h):
        for j in range(grid_w):
            y0, y1 = i * patch_h, (i + 1) * patch_h
            x0, x1 = j * patch_w, (j + 1) * patch_w
            patch = img_rgb[y0:y1, x0:x1]
            ent[i, j] = patch_entropy_gray(patch)

    flat = ent.reshape(-1)
    k = int(np.ceil(top_frac * flat.size))
    k = max(1, k)

    # top-k indices by entropy
    topk_idx = np.argpartition(-flat, k - 1)[:k]
    hard = np.zeros_like(flat, dtype=np.float32)
    hard[topk_idx] = 1.0
    hard_mask = hard.reshape(grid_h, grid_w)

    return ent, hard_mask

def overlay_boundary_mask(
    img_rgb: np.ndarray,
    hard_mask: np.ndarray,
    alpha_red: float = 0.4,
) -> Image.Image:
    """
    Overlay red on boundary patches where hard_mask==1.
    img_rgb: (H,W,3) uint8
    hard_mask: (grid_h, grid_w)
    """
    grid_h, grid_w = hard_mask.shape
    H, W, _ = img_rgb.shape
    patch_h = H // grid_h
    patch_w = W // grid_w

    # Crop to divisible grid (consistent with entropy fn)
    Hc = patch_h * grid_h
    Wc = patch_w * grid_w
    base = img_rgb[:Hc, :Wc].copy()

    out = base.copy()
    red = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
    red[..., 0] = 255

    for i in range(grid_h):
        for j in range(grid_w):
            if hard_mask[i, j] == 1.0:
                y0, y1 = i * patch_h, (i + 1) * patch_h
                x0, x1 = j * patch_w, (j + 1) * patch_w
                patch = out[y0:y1, x0:x1].astype(np.float32)
                out[y0:y1, x0:x1] = ((1 - alpha_red) * patch + alpha_red * red).astype(np.uint8)

    return Image.fromarray(out)

def visualize_entropy_boundaries_single_multi(
    root_dir: str,
    save_path: str,
    grid_h: int = 14,
    grid_w: int = 14,
    top_frac: float = 0.25,
    resize_to: tuple[int, int] = (224, 224),  # e.g. (224, 224)
):
    names = ["single_1", "multi_1", "single_2", "multi_2"]
    # names = ["single_3", "multi_3", "single_4", "multi_4"]

    paths = [os.path.join(root_dir, f"{n}.JPEG") for n in names]

    overlays = []
    entropy_maps = []

    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")

        img = Image.open(p).convert("RGB")

        if resize_to is not None:
            img = img.resize((resize_to[1], resize_to[0]), Image.BILINEAR)

        img_np = np.array(img).astype(np.uint8)

        ent_map, hard_mask = entropy_boundary_mask(
            img_np, grid_h=grid_h, grid_w=grid_w, top_frac=top_frac
        )
        ov = overlay_boundary_mask(img_np, hard_mask)

        overlays.append(ov)
        entropy_maps.append(ent_map)

        print(f"\n== {os.path.basename(p)} ==")
        print(f"entropy map shape: {ent_map.shape}, boundary tokens: {int(hard_mask.sum())}/{hard_mask.size}")
        # If you want to inspect:
        # print(ent_map)
        # print(hard_mask)

    # --- plot overlays 2x2 ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    plot_titles = ["single_1", "multi_1", "Single_2", "multi_2"]

    for ax, ov, title in zip(axes, overlays, plot_titles):
        ax.imshow(ov)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

    return entropy_maps  # handy if you want to visualize heatmaps later



if __name__ == "__main__":
    root_dir = "/users/PAS2912/yusenpeng/Fast-CLIP/unit_further_vis/single_multi"
    save_path = "./entropy_boundary_demo_1st_set.png"

    visualize_entropy_boundaries_single_multi(
        root_dir=root_dir,
        save_path=save_path,
        grid_h=14,
        grid_w=14,
        top_frac=0.25,        # top 25% entropy patches = boundaries
        resize_to=(224, 224), # optional but recommended
    )
