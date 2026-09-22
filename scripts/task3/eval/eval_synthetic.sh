#!/bin/bash
#SBATCH --job-name=PatchOCR_PruneSID_8x
#SBATCH --output=PatchOCR_PruneSID_8x.log
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP_flash
source activate DRIP_flash

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))

cd /users/PAS2912/yusenpeng/DRIP/

VERSION="PruneSID_8x"

DATA_ROOT="/fs/scratch/PAS2836/yusenpeng_dataset/synthetic_eval"

# python src/model_vqa_synthetic_runner.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_8x_pretrain_NEW_DOWN_temp10_train_full \
#     --question-file ${DATA_ROOT}/annotations.jsonl \
#     --image-folder ${DATA_ROOT} \
#     --answers-file ${DATA_ROOT}/answers/${VERSION}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1


python src/model_vqa_synthetic_runner.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_FLASH_finetune_ALL_ONCE_full \
    --question-file ${DATA_ROOT}/annotations.jsonl \
    --image-folder ${DATA_ROOT} \
    --answers-file ${DATA_ROOT}/answers/${VERSION}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1


# python src/model_vqa_synthetic_runner.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Fixed_8x_SCALE_train_full/checkpoint-1200 \
#     --question-file ${DATA_ROOT}/annotations.jsonl \
#     --image-folder ${DATA_ROOT} \
#     --answers-file ${DATA_ROOT}/answers/${VERSION}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1


python src/model_vqa_synthetic_evaluater.py \
    --answers-file ${DATA_ROOT}/answers/${VERSION}.jsonl

conda deactivate