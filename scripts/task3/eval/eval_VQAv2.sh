#!/bin/bash
#SBATCH --job-name=May25_VQAv2_LLaVA_7B_Fixed_100x_finetune_train_full
#SBATCH --output=May25_VQAv2_LLaVA_7B_Fixed_100x_finetune_train_full.log
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP
source activate DRIP

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))

VERSION="LLaVA_7B_Fixed_100x_finetune_train_full"

cd /users/PAS2912/yusenpeng/DRIP/

python src/model_vqa_loader.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Fixed_100x_finetune_train_full \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VQAv2/llava_vqav2_mscoco_test-dev2015.jsonl \
    --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VQAv2/test2015 \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VQAv2/answers/${VERSION}.jsonl \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode vicuna_v1

# python src/model_vqa_loader.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Fixed_100x_finetune_train_lora \
#     --model-base lmsys/vicuna-7b-v1.5 \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VQAv2/llava_vqav2_mscoco_test-dev2015.jsonl \
#     --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VQAv2/test2015 \
#     --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VQAv2/answers/${VERSION}.jsonl \
#     --num-chunks 1 \
#     --chunk-idx 0 \
#     --temperature 0 \
#     --conv-mode vicuna_v1

python src/convert_vqav2_for_submission.py \
    --split llava_vqav2_mscoco_test-dev2015 \
    --ckpt ${VERSION}

conda deactivate
# End of script