#!/bin/bash
#SBATCH --job-name=May25_MME_LLaVA_7B_Fixed_100x_finetune_train_full
#SBATCH --output=May25_MME_LLaVA_7B_Fixed_100x_finetune_train_full.log
#SBATCH --time=00:15:00
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

python src/model_vqa_loader.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Fixed_100x_finetune_train_full \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MME/llava_mme.jsonl \
    --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MME/MME_Benchmark_release_version \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MME/answers/${VERSION}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# python src/model_vqa_loader.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Fixed_100x_finetune_train_lora \
#     --model-base lmsys/vicuna-7b-v1.5 \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MME/llava_mme.jsonl \
#     --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MME/MME_Benchmark_release_version \
#     --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MME/answers/${VERSION}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

cd /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MME

python convert_answer_to_mme.py --experiment ${VERSION}

cd eval_tool

python calculation.py --results_dir answers/${VERSION}

conda deactivate
# End of script