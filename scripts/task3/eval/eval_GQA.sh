#!/bin/bash
#SBATCH --job-name=May25_GQA_LLaVA_7B_Fixed_100x_finetune_train_full
#SBATCH --output=May25_GQA_LLaVA_7B_Fixed_100x_finetune_train_full.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP
source activate DRIP

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))

cd /users/PAS2912/yusenpeng/DRIP/

GQADIR="/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/GQA/data"

OUTPUT_DIR=/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/GQA/answers/llava_gqa_testdev_balanced
mkdir -p $OUTPUT_DIR

OUTPUT_FILE=$OUTPUT_DIR/LLaVA_7B_Fixed_100x_finetune_train_full.jsonl

python src/model_vqa_loader.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Fixed_100x_finetune_train_full \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/GQA/llava_gqa_testdev_balanced.jsonl \
    --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/GQA/data/images \
    --answers-file $OUTPUT_FILE \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode vicuna_v1

# python src/model_vqa_loader.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Fixed_100x_finetune_train_lora \
#     --model-base lmsys/vicuna-7b-v1.5 \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/GQA/llava_gqa_testdev_balanced.jsonl \
#     --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/GQA/data/images \
#     --answers-file $OUTPUT_FILE \
#     --num-chunks 1 \
#     --chunk-idx 0 \
#     --temperature 0 \
#     --conv-mode vicuna_v1

MERGED_FILE=$OUTPUT_DIR/merge.jsonl
cp $OUTPUT_FILE $MERGED_FILE

# Convert for evaluation
python src/convert_gqa_for_eval.py \
    --src $MERGED_FILE \
    --dst /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/GQA/data/testdev_balanced_predictions.json

# Run evaluation
cd /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/GQA/data
python eval/eval.py --tier testdev_balanced

conda deactivate
# End of script