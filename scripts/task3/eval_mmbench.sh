#!/bin/bash
#SBATCH --job-name=FINE_mmb_DRIP
#SBATCH --output=FINE_mmb_DRIP.txt
#SBATCH --time=01:00:00
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

SPLIT="mmbench_dev_20230712"


mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmbench/answers
mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmbench/answers/mmbench_dev_20230712
touch /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmbench/answers/mmbench_dev_20230712/ViTbased-DRIP-4x-16-4-8-finetune-ALL.jsonl

python src/model_vqa_mmbench.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/ViTbased-DRIP-4x-16-4-8-finetune-ALL \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmbench/mmbench_dev_20230712.tsv \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmbench/answers/mmbench_dev_20230712/ViTbased-DRIP-4x-16-4-8-finetune-ALL.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmbench/answers_upload/mmbench_dev_20230712

python src/convert_mmbench_for_submission.py \
    --annotation-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmbench/mmbench_dev_20230712.tsv \
    --result-dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmbench/answers/mmbench_dev_20230712 \
    --upload-dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmbench/answers_upload/mmbench_dev_20230712 \
    --experiment ViTbased-DRIP-4x-16-4-8-finetune-ALL