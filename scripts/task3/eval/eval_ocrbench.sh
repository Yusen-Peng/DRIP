#!/bin/bash
#SBATCH --job-name=May25_OCRBench_LLaVA_7B_PruMerge_8x_pretrain
#SBATCH --output=May25_OCRBench_LLaVA_7B_PruMerge_8x_pretrain.log
#SBATCH --time=00:20:00
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

VERSION="LLaVA_7B_PruMerge_8x_pretrain"


python src/model_vqa_ocrbench.py \
    --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/llava-v1.5-7b-local \
    --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench_Images \
    --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/results \
    --save_name ${VERSION} \
    --OCRBench_file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench.json \
    --temperature 0 \
    --conv_mode vicuna_v1 \
    --num_workers 1

# python src/model_vqa_ocrbench.py \
#     --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/llava-v1.5-7b-lora-local \
#     --model_base lmsys/vicuna-7b-v1.5 \
#     --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench_Images \
#     --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/results \
#     --save_name ${VERSION} \
#     --OCRBench_file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench.json \
#     --temperature 0 \
#     --conv_mode vicuna_v1 \
#     --num_workers 1

conda deactivate
