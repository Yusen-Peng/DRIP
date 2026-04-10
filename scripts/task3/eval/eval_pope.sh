#!/bin/bash
#SBATCH --job-name=Apr9_POPE_reproduce
#SBATCH --output=Apr9_POPE_reproduce.txt
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
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
mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/answers
touch /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/POPE/answers/original_reproduced.jsonl

python src/model_vqa_loader.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/ViTbased-DRIP-4x-16-4-8-finetune-ALL \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/pope/llava_pope_test.jsonl \
    --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/pope/ \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/pope/answers/DRIP-finetune.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python src/eval_pope.py \
    --annotation-dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/pope/anno \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/pope/llava_pope_test.jsonl \
    --result-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/pope/answers/DRIP-finetune.jsonl