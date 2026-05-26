#!/bin/bash
#SBATCH --job-name=May25_MMBench_LLaVA_7B_Fixed_100x_finetune_train_full
#SBATCH --output=May25_MMBench_LLaVA_7B_Fixed_100x_finetune_train_full.log
#SBATCH --time=00:30:00
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

VERSION="LLaVA_7B_Fixed_100x_finetune_train_full"

mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMBench/answers
mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMBench/answers/mmbench_dev_20230712
touch /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMBench/answers/mmbench_dev_20230712/${VERSION}.jsonl

python src/model_vqa_mmbench.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Fixed_100x_finetune_train_full \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMBench/mmbench_dev_20230712.tsv \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMBench/answers/mmbench_dev_20230712/${VERSION}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

# python src/model_vqa_mmbench.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Fixed_100x_finetune_train_lora \
#     --model-base lmsys/vicuna-7b-v1.5 \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMBench/mmbench_dev_20230712.tsv \
#     --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMBench/answers/mmbench_dev_20230712/${VERSION}.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1

mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMBench/answers_upload/mmbench_dev_20230712

python src/convert_mmbench_for_submission.py \
    --annotation-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMBench/mmbench_dev_20230712.tsv \
    --result-dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMBench/answers/mmbench_dev_20230712 \
    --upload-dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMBench/answers_upload/mmbench_dev_20230712 \
    --experiment ${VERSION}

echo "Evaluation completed."