#!/bin/bash
#SBATCH --job-name=0628_SQA_LLaVA_7B_FLASH_finetune_PruneSID_10x_full_second
#SBATCH --output=0628_SQA_LLaVA_7B_FLASH_finetune_PruneSID_10x_full_second.log
#SBATCH --time=0:30:00
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


VERSION="LLaVA_7B_FLASH_finetune_PruneSID_10x_full_second"

cd /users/PAS2912/yusenpeng/DRIP/

mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA/answers
touch /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA/answers/${VERSION}.jsonl

python src/model_vqa_science.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_FLASH_second_to_last_finetune_full \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA/SQA_key.json \
    --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA/images/test \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA/answers/${VERSION}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1


# python src/model_vqa_science_qwen.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_DRIP_10x_train_full \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA/SQA_key.json \
#     --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA/images/test \
#     --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA/answers/${VERSION}.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode qwen_v2


# python src/model_vqa_science.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_10x_second_to_last_train_lora \
#     --model-base lmsys/vicuna-7b-v1.5 \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA/SQA_key.json \
#     --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA/images/test \
#     --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA/answers/${VERSION}.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1

touch /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA/answers/${VERSION}.jsonl
touch /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA/answers/${VERSION}_output.jsonl
touch /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA/answers/${VERSION}_result.json

python src/eval_science_qa.py \
    --base-dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA \
    --result-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA/answers/${VERSION}.jsonl \
    --output-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA/answers/${VERSION}_output.jsonl \
    --output-result /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SQA/answers/${VERSION}_result.json

conda deactivate
# End of script