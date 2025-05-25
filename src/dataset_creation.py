import json
import csv
import random
from pathlib import Path
import pandas as pd
from typing import Tuple
from open_clip_train_local.main import main as train_function


import json
import csv
import random
import shutil
from pathlib import Path
from typing import Tuple

def dataset_processing_COCO_2014(train_samples: int, val_samples: int) -> Tuple[str, str]:
    ANNOTATIONS_TRAIN_FILE = 'annotations/captions_train2014.json'
    ANNOTATIONS_VAL_FILE = 'annotations/captions_val2014.json'
    IMAGES_TRAIN_DIR = Path('train2014')
    IMAGES_VAL_DIR = Path('val2014')
    OUTPUT_TRAIN_CSV = 'dataset/coco_train_subset.csv'
    OUTPUT_VAL_CSV = 'dataset/coco_val_subset.csv'
    OUTPUT_TRAIN_IMG_DIR = Path('dataset/images/train')
    OUTPUT_VAL_IMG_DIR = Path('dataset/images/val')

    OUTPUT_TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_VAL_IMG_DIR.mkdir(parents=True, exist_ok=True)

    def load_and_copy_caption_pairs(annotations_file: str, images_dir: Path, num_samples: int, output_img_dir: Path):
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)

        id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        all_pairs = []

        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            caption = ann['caption'].replace('\n', ' ').strip()
            filename = id_to_filename[img_id]
            src_path = images_dir / filename
            dst_path = output_img_dir / filename

            if src_path.exists():
                all_pairs.append((src_path, dst_path, caption))

        random.seed(42)
        sampled = random.sample(all_pairs, num_samples)

        csv_rows = []
        for src, dst, caption in sampled:
            shutil.copy(src, dst)
            csv_rows.append((str(dst), caption))

        return csv_rows

    # Process and copy images
    train_pairs = load_and_copy_caption_pairs(ANNOTATIONS_TRAIN_FILE, IMAGES_TRAIN_DIR, train_samples, OUTPUT_TRAIN_IMG_DIR)
    val_pairs = load_and_copy_caption_pairs(ANNOTATIONS_VAL_FILE, IMAGES_VAL_DIR, val_samples, OUTPUT_VAL_IMG_DIR)

    # Write CSVs
    with open(OUTPUT_TRAIN_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'caption'])
        writer.writerows(train_pairs)

    with open(OUTPUT_VAL_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'caption'])
        writer.writerows(val_pairs)

    print(f"Wrote {len(train_pairs)} training and {len(val_pairs)} validation samples to CSV.")
    print(f"Copied images to: {OUTPUT_TRAIN_IMG_DIR} and {OUTPUT_VAL_IMG_DIR}")




def downsample_imagenet_val(
    source_dir: str = "ImageNet_val",
    target_dir: str = "dataset/ImageNet_val_subset",
    total_images: int = 5000
):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Collect all image paths
    all_images = list(source_dir.glob("*/*.JPEG"))
    print(f"Found {len(all_images)} total images in {source_dir}")

    if total_images > len(all_images):
        raise ValueError(f"Requested {total_images} images but only {len(all_images)} found.")

    # Sample subset
    sampled = random.sample(all_images, total_images)

    # Copy to new structure
    for path in sampled:
        synset = path.parent.name
        target_class_dir = target_dir / synset
        target_class_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(path, target_class_dir / path.name)

    print(f"✅ Copied {total_images} images into {target_dir.resolve()}")




def main():
    # # Process dataset
    TOTAL_NUM_TRAIN_SAMPLES = 82_783    # ALL
    TOTAL_NUM_VAL_SAMPLES = 40_504      # ALL
    print(f"Processing dataset with {TOTAL_NUM_TRAIN_SAMPLES} training samples and {TOTAL_NUM_VAL_SAMPLES} validation samples...")
    dataset_processing_COCO_2014(TOTAL_NUM_TRAIN_SAMPLES, TOTAL_NUM_VAL_SAMPLES)

    #total_images = 100
    # downsample_imagenet_val(
    #     source_dir="ImageNet_val",
    #     target_dir="dataset/ImageNet_val_subset",
    #     total_images=total_images
    # )


if __name__ == "__main__":
    main()