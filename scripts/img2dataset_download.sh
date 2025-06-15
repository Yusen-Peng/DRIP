#!/bin/bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1    # Also limit other BLAS variants

# -------- CONFIG --------
SHARD_ID="00000"
DATASET_NAME="laion/relaion2B-en-research-safe"
OUTPUT_DIR="/fs/scratch/PAS2836/yusenpeng_dataset/laion_parquet/laion2B-data-shards"
NUM_THREADS=64
COUNT=10000000  # 10M samples
# ------------------------

# 🧠 Authenticate with Hugging Face if needed
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Hugging Face token not set. Please run: export HF_TOKEN=your_token_here"
    exit 1
fi

# Step 1: Install img2dataset if needed
if ! command -v img2dataset &> /dev/null; then
    echo "Installing img2dataset..."
    pip install img2dataset
fi

# Step 2: Find the actual filename for the shard
PARQUET_FILE=$(python -c "
from huggingface_hub import list_repo_files
files = [f for f in list_repo_files('${DATASET_NAME}', repo_type='dataset') if f.startswith('part-${SHARD_ID}') and f.endswith('.parquet')]
print(files[0])
")

PARQUET_URL="https://huggingface.co/datasets/${DATASET_NAME}/resolve/main/${PARQUET_FILE}"
echo "📥 Downloading Parquet shard: $PARQUET_URL"

# Step 3: Download with auth header
wget -nc --header="Authorization: Bearer $HF_TOKEN" -P "$OUTPUT_DIR" "$PARQUET_URL"

# Step 4: Run img2dataset
echo "🚀 Starting img2dataset..."
img2dataset \
    --url_list "$OUTPUT_DIR/$PARQUET_FILE" \
    --input_format "parquet" \
    --url_col "url" \
    --caption_col "caption" \
    --output_format "webdataset" \
    --output_folder "$OUTPUT_DIR" \
    --processes_count $NUM_THREADS \
    --thread_count 4 \
    --number_sample_per_shard 10000 \
    --image_size 256 \
    --resize_mode "no" \
    --enable_wandb False \
    --disallowed_header_directives "User-Agent" \
    --save_additional_columns '["similarity"]' \
    --count $COUNT

echo "✅ Done. Output stored in: $OUTPUT_DIR"

REPORT_FILE="$OUTPUT_DIR/report.json"

if [ -f "$REPORT_FILE" ]; then
    echo "🧾 Parsing report..."
    num_processed=$(jq .num_processed "$REPORT_FILE")
    num_success=$(jq .num_success "$REPORT_FILE")
    num_failed=$(jq .num_failed "$REPORT_FILE")

    echo "🧮 Processed: $num_processed" | tee -a "$OUTPUT_DIR/download_summary.log"
    echo "📸 Successfully downloaded: $num_success" | tee -a "$OUTPUT_DIR/download_summary.log"
    echo "❌ Failed downloads: $num_failed" | tee -a "$OUTPUT_DIR/download_summary.log"
else
    echo "⚠️  No report.json found in $OUTPUT_DIR" | tee -a "$OUTPUT_DIR/download_summary.log"
fi
