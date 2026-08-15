#!/bin/bash
#SBATCH --job-name=0812_POPE_LLaVA_7B_Fixed_4x_SCALE_train_full-checkpoint-600
#SBATCH --output=0812_POPE_LLaVA_7B_Fixed_4x_SCALE_train_full-checkpoint-600.log
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP_flash
source activate DRIP_flash

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))

VERSION="LLaVA_7B_Fixed_4x_SCALE_train_full-checkpoint-600"

cd /users/PAS2912/yusenpeng/DRIP/
mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/answers
touch /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/answers/${VERSION}.jsonl


python src/model_vqa_loader.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Fixed_4x_SCALE_train_full/checkpoint-600 \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/llava_pope_test.jsonl \
    --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/val2014 \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/answers/${VERSION}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# python src/model_vqa_loader.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_4x_temp15_new_downsample_train_full \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/llava_pope_test.jsonl \
#     --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/val2014 \
#     --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/answers/${VERSION}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python src/model_vqa_loader_qwen.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_DRIP_4x_train_full \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/llava_pope_test.jsonl \
#     --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/val2014 \
#     --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/answers/${VERSION}.jsonl \
#     --temperature 0 \
#     --conv-mode qwen_v2

# python src/model_vqa_loader.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_FLASH_second_to_last_finetune_lora \
#     --model-base lmsys/vicuna-7b-v1.5 \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/llava_pope_test.jsonl \
#     --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/val2014 \
#     --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/answers/${VERSION}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

python src/eval_pope.py \
    --annotation-dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/anno \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/llava_pope_test.jsonl \
    --result-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/answers/${VERSION}.jsonl

conda deactivate
# End of script
