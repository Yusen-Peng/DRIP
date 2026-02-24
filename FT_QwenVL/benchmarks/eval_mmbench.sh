#!/bin/bash
#SBATCH --job-name=Nov20_fixed_pooling
#SBATCH --output=Nov20_fixed_pooling.txt
#SBATCH --time=00:15:00
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

cd /users/PAS2912/yusenpeng/Fast-CLIP/

# SPLIT="mmbench_dev_20230712"


mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmbench/answers
mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmbench/answers/mmbench_dev_20230712
touch /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmbench/answers/mmbench_dev_20230712/fixed-pooling-finetune.jsonl

python src/model_vqa_mmbench.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/fixed-pooling-finetune \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmbench/mmbench_dev_20230712.tsv \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmbench/answers/mmbench_dev_20230712/fixed-pooling-finetune.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode llava_v1 \

mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmbench/answers_upload/mmbench_dev_20230712

python src/convert_mmbench_for_submission.py \
    --annotation-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmbench/mmbench_dev_20230712.tsv \
    --result-dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmbench/answers/mmbench_dev_20230712 \
    --upload-dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmbench/answers_upload/mmbench_dev_20230712 \
    --experiment fixed-pooling-finetune
echo "Evaluation completed."