import os
import json
import random
import argparse

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 336

# CLIP-L/14 patch size
PATCH_SIZE = 14

# Synthetic checkerboard cell size
# 56 px = 4 x 4 CLIP patches
CELL_SIZE = 56

# 336 / 56 = 6
GRID_SIZE = IMAGE_SIZE // CELL_SIZE

# One character per quadrant:
# top-left, top-right, bottom-left, bottom-right
NUM_CHARACTERS = 4
NUM_QUESTIONS_PER_IMAGE = 4

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


"""
Example:

python src/synthetic_eval/dataset_pipeline.py \
    --output-dir /fs/scratch/PAS2836/yusenpeng_dataset/synthetic_eval \
    --num-images 100 \
    --font-path /usr/share/fonts/dejavu-sans-fonts/DejaVuSansCondensed-Bold.ttf \
    --font-size 44
"""


# ============================================================
# Font
# ============================================================

def load_font(font_path=None, font_size=44):

    if font_path is not None:
        return ImageFont.truetype(
            font_path,
            font_size,
        )

    return ImageFont.load_default()


# ============================================================
# Checkerboard
# ============================================================

def make_checkerboard():
    """
    Create a 336x336 checkerboard consisting of 6x6 cells.

    Each cell:
        56 x 56 pixels
        4 x 4 CLIP patches
        16 CLIP visual tokens

    Total:
        6 x 6 = 36 cells
        24 x 24 = 576 CLIP patches
    """

    image = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        "white",
    )

    draw = ImageDraw.Draw(image)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):

            color = (
                "white"
                if (row + col) % 2 == 0
                else "black"
            )

            x0 = col * CELL_SIZE
            y0 = row * CELL_SIZE

            x1 = x0 + CELL_SIZE
            y1 = y0 + CELL_SIZE

            draw.rectangle(
                [
                    x0,
                    y0,
                    x1 - 1,
                    y1 - 1,
                ],
                fill=color,
            )

    return image


# ============================================================
# Character rendering
# ============================================================

def draw_centered_character(
    draw,
    row,
    col,
    char,
    font,
):
    """
    Draw a character centered inside one checkerboard cell.

    White cell -> black character
    Black cell -> white character
    """

    white_cell = (row + col) % 2 == 0

    text_color = (
        "black"
        if white_cell
        else "white"
    )

    x0 = col * CELL_SIZE
    y0 = row * CELL_SIZE

    # Determine actual rendered character bounds
    bbox = draw.textbbox(
        (0, 0),
        char,
        font=font,
    )

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Center character inside cell
    x = (
        x0
        + (CELL_SIZE - text_w) / 2
        - bbox[0]
    )

    y = (
        y0
        + (CELL_SIZE - text_h) / 2
        - bbox[1]
    )

    draw.text(
        (x, y),
        char,
        fill=text_color,
        font=font,
    )


# ============================================================
# Generate one image
# ============================================================

def generate_image(font, rng):
    """
    Generate one synthetic checkerboard image.

    Exactly one character is randomly placed inside each
    of the four image quadrants:

        top-left     | top-right
        -----------------------
        bottom-left  | bottom-right

    For a 6x6 grid:

        TL: rows 0-2, cols 0-2
        TR: rows 0-2, cols 3-5
        BL: rows 3-5, cols 0-2
        BR: rows 3-5, cols 3-5

    Returns:
        image
        annotations
    """

    image = make_checkerboard()
    draw = ImageDraw.Draw(image)

    half = GRID_SIZE // 2  # 3

    quadrants = {
        "top-left": [
            (row, col)
            for row in range(0, half)
            for col in range(0, half)
        ],

        "top-right": [
            (row, col)
            for row in range(0, half)
            for col in range(half, GRID_SIZE)
        ],

        "bottom-left": [
            (row, col)
            for row in range(half, GRID_SIZE)
            for col in range(0, half)
        ],

        "bottom-right": [
            (row, col)
            for row in range(half, GRID_SIZE)
            for col in range(half, GRID_SIZE)
        ],
    }

    annotations = []

    # Exactly one character per quadrant
    for region, cells in quadrants.items():

        # Random location within quadrant
        row, col = rng.choice(cells)

        # Random character
        char = rng.choice(CHARSET)

        draw_centered_character(
            draw=draw,
            row=row,
            col=col,
            char=char,
            font=font,
        )

        annotations.append({
            "region": region,

            # Keep exact cell coordinates for later analysis.
            # Human-readable 1-indexed coordinates.
            "row": row + 1,
            "column": col + 1,

            "character": char,
        })

    return image, annotations


# ============================================================
# Dataset generation
# ============================================================

def generate_dataset(
    output_dir,
    num_images,
    seed=42,
    font_path=None,
    font_size=44,
):

    rng = random.Random(seed)

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    image_dir = os.path.join(
        output_dir,
        "images",
    )

    os.makedirs(
        image_dir,
        exist_ok=True,
    )

    annotation_path = os.path.join(
        output_dir,
        "annotations.jsonl",
    )

    # --------------------------------------------------------
    # Font
    # --------------------------------------------------------

    font = load_font(
        font_path=font_path,
        font_size=font_size,
    )

    # --------------------------------------------------------
    # Generate images + questions
    # --------------------------------------------------------

    with open(annotation_path, "w") as f:

        for image_idx in range(num_images):

            image, annotations = generate_image(
                font=font,
                rng=rng,
            )

            # ------------------------------------------------
            # Save image
            # ------------------------------------------------

            image_name = f"{image_idx:06d}.png"

            image_path = os.path.join(
                image_dir,
                image_name,
            )

            image.save(image_path)

            # ------------------------------------------------
            # Generate four directional QA pairs
            # ------------------------------------------------

            for ann in annotations:

                region = ann["region"]
                row = ann["row"]
                col = ann["column"]
                char = ann["character"]

                sample = {
                    "id": (
                        f"{image_idx:06d}_"
                        f"{region.replace('-', '_')}"
                    ),

                    "image": (
                        f"images/{image_name}"
                    ),

                    "question": (
                        f"What character is in the "
                        f"{region} part of the image? "
                        f"Answer with a single character."
                    ),

                    "answer": char,

                    # Directional location used by the question
                    "region": region,

                    # Exact coordinates are NOT shown to model.
                    # We retain them for analysis.
                    "row": row,
                    "column": col,
                }

                f.write(
                    json.dumps(sample) + "\n"
                )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("=" * 60)
    print("Synthetic Patch OCR Dataset")
    print("=" * 60)

    print(
        f"Generated images:   {num_images}"
    )

    print(
        f"Characters/image:   {NUM_CHARACTERS}"
    )

    print(
        f"Questions/image:    {NUM_QUESTIONS_PER_IMAGE}"
    )

    print(
        f"Total QA pairs:     "
        f"{num_images * NUM_QUESTIONS_PER_IMAGE}"
    )

    print(
        f"Image resolution:   "
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"Checkerboard:       "
        f"{GRID_SIZE}x{GRID_SIZE}"
    )

    print(
        f"Cell size:          "
        f"{CELL_SIZE}x{CELL_SIZE} px"
    )

    print(
        f"CLIP patches/cell:  "
        f"{CELL_SIZE // PATCH_SIZE}x"
        f"{CELL_SIZE // PATCH_SIZE} "
        f"= {(CELL_SIZE // PATCH_SIZE) ** 2}"
    )

    print(
        f"Font size:          {font_size}"
    )

    print(
        f"Random seed:        {seed}"
    )

    print("-" * 60)

    print(
        f"Images:      {image_dir}"
    )

    print(
        f"Annotations: {annotation_path}"
    )

    print("=" * 60)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--font-path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--font-size",
        type=int,
        default=44,
    )

    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        num_images=args.num_images,
        seed=args.seed,
        font_path=args.font_path,
        font_size=args.font_size,
    )