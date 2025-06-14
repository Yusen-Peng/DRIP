#!/bin/bash

# === CONFIG ===
IMAGENET_ROOT="/fs/scratch/PAS2836/yusenpeng_dataset/"
TRAIN_TAR="$IMAGENET_ROOT/ILSVRC2012_img_train.tar"
VAL_TAR="$IMAGENET_ROOT/ILSVRC2012_img_val.tar"
DEVKIT_TAR="$IMAGENET_ROOT/ILSVRC2012_devkit_t12.tar.gz"

# === Step 1: Extract training set ===
echo "📦 Extracting training images..."
mkdir -p "$IMAGENET_ROOT/train"
cd "$IMAGENET_ROOT/train"
tar -xf "$TRAIN_TAR"

echo "📂 Organizing training images into folders..."
for archive in *.tar; do
    class_name="${archive%.tar}"
    mkdir -p "$class_name"
    tar -xf "$archive" -C "$class_name"
    rm "$archive"
done
echo "✅ Training set organized."

# === Step 2: Extract validation set ===
echo "📦 Extracting validation images..."
mkdir -p "$IMAGENET_ROOT/val"
cd "$IMAGENET_ROOT/val"
tar -xf "$VAL_TAR"

# === Step 3: Extract devkit for val labels ===
echo "📦 Extracting devkit..."
cd "$IMAGENET_ROOT"
tar -xf "$DEVKIT_TAR"
LABELS_FILE=$(find . -name 'ILSVRC2012_validation_ground_truth.txt')

# === Step 4: Map val images to folders using ground truth ===
echo "📂 Organizing validation images..."
cd "$IMAGENET_ROOT/val"
mkdir -p temp
mv *.JPEG temp/
cd temp

INDEX=0
while read -r CLASS_ID; do
    INDEX=$((INDEX + 1))
    FILE_NAME=$(printf "ILSVRC2012_val_%08d.JPEG" $INDEX)
    CLASS_DIR="../n$CLASS_ID"
    mkdir -p "$CLASS_DIR"
    mv "$FILE_NAME" "$CLASS_DIR/"
done < "$LABELS_FILE"

cd ..
rm -r temp
echo "✅ Validation set organized."

echo "🏁 ImageNet preparation complete!"