#!/bin/bash
#SBATCH --job-name=May5_VQAv2_LLaVA_7B_LoRA_checkpoint
#SBATCH --output=May5_VQAv2_LLaVA_7B_LoRA_checkpoint.log
#SBATCH --time=07:00:00
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

VERSION="LLaVA_7B_LoRA_checkpoint"

cd /users/PAS2912/yusenpeng/DRIP/

# python src/model_vqa_loader.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/llava-v1.5-7b-local \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VQAv2/llava_vqav2_mscoco_test-dev2015.jsonl \
#     --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VQAv2/test2015 \
#     --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VQAv2/answers/${VERSION}.jsonl \
#     --num-chunks 1 \
#     --chunk-idx 0 \
#     --temperature 0 \
#     --conv-mode vicuna_v1

python src/model_vqa_loader.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/llava-v1.5-7b-lora-local \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VQAv2/llava_vqav2_mscoco_test-dev2015.jsonl \
    --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VQAv2/test2015 \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VQAv2/answers/${VERSION}.jsonl \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode vicuna_v1

python src/convert_vqav2_for_submission.py \
    --split llava_vqav2_mscoco_test-dev2015 \
    --ckpt ${VERSION}

conda deactivate
# End of script