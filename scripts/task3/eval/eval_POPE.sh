#!/bin/bash
#SBATCH --job-name=0702_POPE_LLaVA_7B_Perceiver_8x_train_all
#SBATCH --output=0702_POPE_LLaVA_7B_Perceiver_8x_train_all.log
#SBATCH --time=00:50:00
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

VERSION="LLaVA_7B_Perceiver_8x_train_all"

cd /users/PAS2912/yusenpeng/DRIP/
mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/answers
touch /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/answers/${VERSION}.jsonl


python src/model_vqa_loader.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Perceiver_8x_train_all \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/llava_pope_test.jsonl \
    --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/val2014 \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/answers/${VERSION}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# python src/model_vqa_loader_qwen.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_DRIP_10x_train_full \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/llava_pope_test.jsonl \
#     --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/val2014 \
#     --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/answers/${VERSION}.jsonl \
#     --temperature 0 \
#     --conv-mode qwen_v2

# python src/model_vqa_loader.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_10x_second_to_last_train_lora \
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
