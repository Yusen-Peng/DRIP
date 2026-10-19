#!/bin/bash
#SBATCH --job-name=MMDU_LLaVA_7B_DRIP_8x
#SBATCH --output=MMDU_LLaVA_7B_DRIP_8x.log
#SBATCH --time=01:00:00
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

set -a
source /users/PAS2912/yusenpeng/DRIP/.env
set +a

# VERSION="LLaVA_7B_ORIGINAL"
VERSION="LLaVA_7B_DRIP_8x"

mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMDU/save_answer/${VERSION}

# python src/model_vqa_mmdu.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_8x_pretrain_NEW_DOWN_temp10_train_full \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMDU/benchmark.json \
#     --answers-dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMDU/save_answer/${VERSION} \
#     --temperature 0 \
#     --conv-mode vicuna_v1

python src/MMDU_eval/prompt.py \
    --answers-dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMDU/save_answer/${VERSION} \
    --scores-dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMDU/save_score/${VERSION} \

python src/MMDU_eval/statistics.py \
    --folder-path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMDU/save_score/${VERSION}

conda deactivate
# End of script