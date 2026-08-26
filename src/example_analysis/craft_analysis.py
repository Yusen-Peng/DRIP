import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from datasets import load_dataset
from tqdm import tqdm
from hezar.models import Model as CRAFTModel
from hezar.utils import load_image

PROJECT_ROOT = "/users/PAS2912/yusenpeng/DRIP"
sys.path.insert(0, PROJECT_ROOT)
from src.boundary_visual_LLaVA import load_img_with_processor, overlay_llava_drip_boundaries, build_llava_drip_vision_tower, build_llava_siglip_vision_tower


"""
How to run this script:

salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --partition debug-nextgen --time 00:30:00
module load miniconda3/24.1.2-py310
conda activate DRIP
python src/example_analysis/craft_analysis.py
"""


"""
    showing a few examples for the paper.
"""
# SAVE_VISUALIZATIONS = True
# IMAGE_DIR = "/users/PAS2912/yusenpeng/DRIP/src/example_analysis/TextVQA_results/subset_images"
# OUTPUT_DIR = "/users/PAS2912/yusenpeng/DRIP/src/example_analysis/text_boundary_overlap_subset"


BENCHMARK = "OCRBenchv2" # TextVQA, OCRBench, OCRBenchv2, DocVQA, ChartQAPro



"""
    performing evaluation on the whole OCR benchmarks
"""
SAVE_VISUALIZATIONS = False

if BENCHMARK == "TextVQA":
    IMAGE_DIR = Path("/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/train_images")
elif BENCHMARK == "OCRBench":
    IMAGE_DIR = Path("/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench_Images")
elif BENCHMARK == "DocVQA":
    IMAGE_DIR = Path("/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/images")
elif BENCHMARK == "OCRBenchv2":
    IMAGE_DIR = None
elif BENCHMARK == "ChartQAPro":
    IMAGE_DIR = Path("/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/images")
else:
    raise ValueError(f"Unknown benchmark: {BENCHMARK}")


OUTPUT_DIR = f"/users/PAS2912/yusenpeng/DRIP/src/example_analysis/text_boundary_overlap_{BENCHMARK}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_finetune_train_full/drip.bin"
# NOTE: new downsample function version
# DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_pretrain_NEW_DOWN_temp001_train_full/drip.bin"
# COMPRESSION_RATE = 0.25


# DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_8x_finetune_train_full/drip.bin"
# NOTE: new downsample function version
# DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_8x_pretrain_NEW_DOWN_temp10_train_full/drip.bin"
# COMPRESSION_RATE = 0.125


# DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_10x_finetune_train_full/drip.bin"
# NOTE: new downsample function version
DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_10x_pretrain_NEW_DOWN_temp10_train_full/drip.bin"
COMPRESSION_RATE = 0.1



MERGE_STRATEGY = "DRIP"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"craft_boundary_overlap_{COMPRESSION_RATE}.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, f"craft_boundary_overlap_summary_{COMPRESSION_RATE}.csv")

VISION_TOWER_NAME = "openai/clip-vit-large-patch14-336"

# For SigLIP:
# VISION_TOWER_NAME = "google/siglip2-large-patch16-384"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")



def load_pil_with_processor(pil_image, image_processor):
    """Same idea as load_img_with_processor(), but accepts a PIL image instead of a filesystem path."""
    pil_image = pil_image.convert("RGB")
    processed = image_processor(images=pil_image, return_tensors="pt")
    return processed["pixel_values"][0]

def processor_tensor_to_pil(img_3chw: torch.Tensor, image_processor):
    """Convert the normalized images tensor produced by the LLaVA image processor back into an RGB PIL image."""
    x = img_3chw.detach().float().cpu()
    mean = getattr(image_processor,"image_mean",[0.5, 0.5, 0.5])
    std = getattr(image_processor,"image_std",[0.5, 0.5, 0.5])

    mean = torch.tensor(mean,dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor(std,dtype=x.dtype).view(3, 1, 1)

    x = x * std + mean
    x = x.clamp(0, 1)
    x = x.permute(1, 2, 0).numpy()
    x = (x * 255.0).round().astype(np.uint8)
    return Image.fromarray(x).convert("RGB")


def get_fixed_hard_mask(model):
    grid_h = model.num_patches_per_side
    grid_w = model.num_patches_per_side
    L_patch = grid_h * grid_w
    num_tokens_to_keep = max(1, int(L_patch * model.compression_rate))
    indices = torch.linspace(0,L_patch - 1,steps=num_tokens_to_keep).round().long()
    hard_boundaries = torch.zeros(L_patch, dtype=torch.float32)
    hard_boundaries[indices] = 1.0
    hard_boundaries[-1] = 1.0
    hard_mask = hard_boundaries.view(grid_h, grid_w).numpy()
    return hard_mask


def patch_mask_to_pixel_mask(hard_mask, image_h, image_w):
    hard_mask = np.asarray(hard_mask)
    grid_h, grid_w = hard_mask.shape
    pixel_mask = np.zeros((image_h, image_w),dtype=bool)

    y_edges = np.linspace(0, image_h, grid_h + 1).round().astype(int)
    x_edges = np.linspace(0, image_w, grid_w + 1).round().astype(int)

    for i in range(grid_h):
        for j in range(grid_w):
            if hard_mask[i, j] <= 0:
                continue
            y0 = y_edges[i]
            y1 = y_edges[i + 1]
            x0 = x_edges[j]
            x1 = x_edges[j + 1]
            pixel_mask[y0:y1, x0:x1] = True

    return pixel_mask


def run_craft_on_processed_image(craft_model: CRAFTModel, processed_pil: Image.Image):
    """We save a temporary PNG because load_image() is known to work with paths in your existing CRAFT setup."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        processed_pil.save(tmp_path)
        craft_image = load_image(tmp_path)
        outputs = craft_model.predict(craft_image)
        boxes = outputs[0]["boxes"]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return boxes


def craft_boxes_to_pixel_mask(boxes, image_h, image_w):
    """CRAFT boxes from hezar: (x, y, width, height)"""
    pil_mask = Image.new("L", (image_w, image_h), 0)
    draw = ImageDraw.Draw(pil_mask)

    for box in boxes:
        x, y, w, h = map(float, box)
        # we need to skip malformed boxes
        if w <= 0 or h <= 0:
            continue

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(image_w, x + w)
        y2 = min(image_h, y + h)

        draw.rectangle(
            [x1, y1, x2, y2],
            fill=1,
        )
    return np.asarray(pil_mask, dtype=bool)



def compute_overlap_metrics(
    boundary_mask,
    text_mask,
):
    """
        boundary_mask: union of selected DRIP / Fixed patches
        text_mask: union of CRAFT text detections

    Metrics
    -------

    IoU:

        |boundary ∩ text|
        -----------------
        |boundary ∪ text|

    text_coverage:

        |boundary ∩ text|
        -----------------
        |text|

    """

    boundary_mask = boundary_mask.astype(bool)
    text_mask = text_mask.astype(bool)

    intersection = np.logical_and(
        boundary_mask,
        text_mask,
    ).sum()

    union = np.logical_or(
        boundary_mask,
        text_mask,
    ).sum()

    boundary_area = boundary_mask.sum()
    text_area = text_mask.sum()

    if text_area == 0:
        # edge case: no text detected by CRAFT, so we cannot compute text coverage

        return {
            "intersection_pixels": np.nan,
            "union_pixels": np.nan,
            "boundary_pixels": boundary_area,
            "text_pixels": 0,

            "iou": np.nan,
            "text_coverage": np.nan,
        }

    iou = (
        intersection / union
        if union > 0
        else np.nan
    )

    text_coverage = (
        intersection / text_area
        if text_area > 0
        else np.nan
    )

    return {
        "intersection_pixels": int(intersection),
        "union_pixels": int(union),
        "boundary_pixels": int(boundary_area),
        "text_pixels": int(text_area),
        "iou": float(iou),
        "text_coverage": float(text_coverage)
    }


def make_overlap_visualization(
    processed_pil,
    boundary_mask,
    boxes,
    save_path,
    alpha=0.35,
):
    """
    Visualization:
        RED   = selected DRIP/Fixed patch
        BLUE  = CRAFT bounding box
    """

    image = np.array(processed_pil).astype(np.float32)

    overlay = image.copy()

    red = np.zeros_like(overlay)
    red[..., 0] = 255

    mask = boundary_mask[..., None]

    overlay = np.where(
        mask,
        (1 - alpha) * overlay + alpha * red,
        overlay,
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    result = Image.fromarray(overlay)

    draw = ImageDraw.Draw(result)
    image_w, image_h = processed_pil.size
    for box in boxes:

        x, y, w, h = map(float, box)

        # again, skip malformed boxes
        if w <= 0 or h <= 0:
            continue

        x1 = x
        y1 = y

        x2 = x + w
        y2 = y + h

        x1 = max(0, min(image_w, x1))
        x2 = max(0, min(image_w, x2))
        y1 = max(0, min(image_h, y1))
        y2 = max(0, min(image_h, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            outline=(255, 255, 0),
            width=2,
        )

    save_dir = os.path.dirname(save_path)

    if save_dir:
        os.makedirs(save_dir,exist_ok=True)

    result.save(save_path)


@torch.no_grad()
def process_image(image_path, vision_model, craft_model, image_is_pil=False):
    if image_is_pil:
        img_tensor = load_pil_with_processor(image_path, vision_model.image_processor)
    else:
        img_tensor = load_img_with_processor(image_path, vision_model.image_processor)

    processed_pil = processor_tensor_to_pil(img_tensor, vision_model.image_processor)

    image_w, image_h = processed_pil.size

    craft_boxes = run_craft_on_processed_image(craft_model,processed_pil)

    text_pixel_mask = craft_boxes_to_pixel_mask(craft_boxes, image_h=image_h, image_w=image_w)

    (
        _,
        drip_hard_mask,
        _,
        drip_num_boundaries,
    ) = overlay_llava_drip_boundaries(vision_model, img_tensor, alpha=0.4)

    drip_pixel_mask = patch_mask_to_pixel_mask(drip_hard_mask, image_h=image_h, image_w=image_w)

    fixed_hard_mask = get_fixed_hard_mask(vision_model)

    fixed_num_boundaries = int(fixed_hard_mask.sum())

    fixed_pixel_mask = patch_mask_to_pixel_mask(fixed_hard_mask, image_h=image_h, image_w=image_w)

    drip_metrics = compute_overlap_metrics(drip_pixel_mask, text_pixel_mask)

    fixed_metrics = compute_overlap_metrics(fixed_pixel_mask, text_pixel_mask)

    return {
        "processed_pil": processed_pil,

        "craft_boxes": craft_boxes,
        "text_pixel_mask": text_pixel_mask,

        "drip_hard_mask": drip_hard_mask,
        "drip_pixel_mask": drip_pixel_mask,
        "drip_num_boundaries": int(drip_num_boundaries),
        "drip_metrics": drip_metrics,

        "fixed_hard_mask": fixed_hard_mask,
        "fixed_pixel_mask": fixed_pixel_mask,
        "fixed_num_boundaries": fixed_num_boundaries,
        "fixed_metrics": fixed_metrics,
    }



def build_summary(df):
    summary_rows = []
    for method in ["DRIP", "Fixed"]:
        valid = df[
            (df["method"] == method)
            & (df["num_craft_boxes"] > 0)
        ]
        summary_rows.append(
            {
                "method": method,
                "num_images": len(valid),
                "mean_iou":
                    valid["iou"].mean(),
                "std_iou":
                    valid["iou"].std(),
                "mean_text_coverage":
                    valid["text_coverage"].mean(),
            }
        )

    return pd.DataFrame(summary_rows)


def find_images_recursive(root_dir):
    """
    Recursively find all images under root_dir.

    Example:
        OCRBench_Images/docVQA/val/documents/foo.png
        OCRBench_Images/ctw/bar.jpg
        ...
    """

    root_dir = Path(root_dir)

    image_paths = sorted(
        p
        for p in root_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return image_paths



def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    if "siglip" in VISION_TOWER_NAME.lower():

        vision_model = build_llava_siglip_vision_tower(
            vision_tower_name=VISION_TOWER_NAME,
            mm_vision_select_layer=-1,
            mm_vision_select_feature="patch",
            compression_rate=COMPRESSION_RATE,
            drip_weight_path=DRIP_WEIGHT_PATH,
            merge_strategy=MERGE_STRATEGY,
            device=DEVICE,
        )
    else:
        vision_model = build_llava_drip_vision_tower(
            vision_tower_name=VISION_TOWER_NAME,
            mm_vision_select_layer=-1,
            mm_vision_select_feature="patch",
            compression_rate=COMPRESSION_RATE,
            drip_weight_path=DRIP_WEIGHT_PATH,
            merge_strategy=MERGE_STRATEGY,
            device=DEVICE,
        )

    print("Vision processor output size:", vision_model.image_processor.size)

    print(
        "Patch grid:",
        vision_model.num_patches_per_side,
        "x",
        vision_model.num_patches_per_side,
    )

    print("Loading CRAFT...")

    craft_model = CRAFTModel.load(
        "hezarai/CRAFT",
        device=DEVICE,
    )

    rows = []


    if BENCHMARK != "OCRBenchv2":
        image_paths = find_images_recursive(IMAGE_DIR)
        print(f"🥶🥶🥶🥶 Found {len(image_paths)} images recursively. 🥶🥶🥶🥶")


        for image_path in tqdm(
            image_paths,
            desc="CRAFT vs boundaries",
        ):

            result = process_image(image_path=str(image_path), vision_model=vision_model, craft_model=craft_model)

            # filename = os.path.basename(image_path)
            relative_path = image_path.relative_to(IMAGE_DIR)
            filename = str(relative_path)

            num_boxes = len(result["craft_boxes"])

            drip_row = {
                "image": filename,
                "method": "DRIP",
                "image_width": result["processed_pil"].width,
                "image_height": result["processed_pil"].height,
                "num_craft_boxes": num_boxes,
                "num_boundary_patches":result["drip_num_boundaries"],
            }

            drip_row.update(result["drip_metrics"])

            rows.append(drip_row)

            fixed_row = {
                "image": filename,
                "method": "Fixed",
                "image_width": result["processed_pil"].width,
                "image_height": result["processed_pil"].height,
                "num_craft_boxes": num_boxes,
                "num_boundary_patches": result["fixed_num_boundaries"]
            }

            fixed_row.update(result["fixed_metrics"])
            rows.append(fixed_row)

            if SAVE_VISUALIZATIONS:

                stem = Path(filename).stem

                drip_save = os.path.join(OUTPUT_DIR,"visualizations","drip",f"{stem}.png")

                fixed_save = os.path.join(OUTPUT_DIR,"visualizations","fixed",f"{stem}.png")

                make_overlap_visualization(
                    processed_pil=result["processed_pil"],
                    boundary_mask=result["drip_pixel_mask"],
                    boxes=result["craft_boxes"],
                    save_path=drip_save
                )

                make_overlap_visualization(
                    processed_pil=result["processed_pil"],
                    boundary_mask=result["fixed_pixel_mask"],
                    boxes=result["craft_boxes"],
                    save_path=fixed_save
                )

    else:
        dataset = load_dataset("lmms-lab/OCRBench-v2", split="test")
        print(f"🥶🥶🥶🥶 Found {len(dataset)} OCRBench-v2 samples. 🥶🥶🥶🥶")

        for sample in tqdm(
            dataset,
            desc="OCRBench-v2 CRAFT vs boundaries",
        ):

            pil_image = sample["image"]

            result = process_image(
                image_path=pil_image,
                vision_model=vision_model,
                craft_model=craft_model,
                image_is_pil=True,
            )

            sample_id = sample["id"]
            dataset_name = sample["dataset_name"]
            question_type = sample["type"]

            num_boxes = len(result["craft_boxes"])

            drip_row = {
                "image": sample_id,
                "dataset_name": dataset_name,
                "question_type": question_type,
                "method": "DRIP",
                "image_width": result["processed_pil"].width,
                "image_height": result["processed_pil"].height,
                "num_craft_boxes": num_boxes,
                "num_boundary_patches": result["drip_num_boundaries"],
            }

            drip_row.update(result["drip_metrics"])
            rows.append(drip_row)

            fixed_row = {
                "image": sample_id,
                "dataset_name": dataset_name,
                "question_type": question_type,
                "method": "Fixed",
                "image_width": result["processed_pil"].width,
                "image_height": result["processed_pil"].height,
                "num_craft_boxes": num_boxes,
                "num_boundary_patches": result["fixed_num_boundaries"],
            }

            fixed_row.update(result["fixed_metrics"])
            rows.append(fixed_row)



    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(f"\nSaved per-image results to:\n{OUTPUT_CSV}")
    summary_df = build_summary(df)

    summary_df.to_csv(SUMMARY_CSV,index=False)

    print("\n==============================")

    print(
        "OVERLAP SUMMARY"
    )

    print(
        "=============================="
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved summary to:\n{SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()
