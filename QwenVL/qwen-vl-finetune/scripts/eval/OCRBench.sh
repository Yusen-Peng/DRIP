#!/bin/bash
#SBATCH --job-name=0826_OCRBench_Qwen3VL_4B_checkpoint_18
#SBATCH --output=0826_OCRBench_Qwen3VL_4B_checkpoint_18.log
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP_qwenvl_flash

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))

cd /users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune/qwenvl

VERSION="0826_OCRBench_Qwen3VL_4B_checkpoint_18"

# python eval_code/model_vqa_ocrbench.py \
#     --model_path Qwen/Qwen3-VL-4B-Instruct \
#     --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench_Images \
#     --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/results \
#     --save_name ${VERSION} \
#     --OCRBench_file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench.json \
#     --temperature 0 \
#     --num_workers 1


python eval_code/model_vqa_ocrbench.py \
    --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL/checkpoint-18 \
    --model_base Qwen/Qwen3-VL-4B-Instruct \
    --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench_Images \
    --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/results \
    --save_name ${VERSION} \
    --OCRBench_file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench.json \
    --temperature 0 \
    --num_workers 1


conda deactivate
