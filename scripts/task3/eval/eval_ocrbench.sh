#!/bin/bash
#SBATCH --job-name=0821_OCRBench_LLaVA_Qwen2.5-14B-Instruct_DRIP_10x_pretrain_NEW_DOWN_temp10_train_full
#SBATCH --output=0821_OCRBench_LLaVA_Qwen2.5-14B-Instruct_DRIP_10x_pretrain_NEW_DOWN_temp10_train_full.log
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=nextgen
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

VERSION="LLaVA_Qwen2.5-14B-Instruct_DRIP_10x_pretrain_NEW_DOWN_temp10_train_full"

# python src/model_vqa_ocrbench.py \
#     --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_10x_pretrain_NEW_DOWN_temp10_train_full \
#     --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench_Images \
#     --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/results \
#     --save_name ${VERSION} \
#     --OCRBench_file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench.json \
#     --temperature 0 \
#     --conv_mode vicuna_v1 \
#     --num_workers 1

# python src/model_vqa_ocrbench.py \
#     --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_10x_temp10_new_downsample_train_full \
#     --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench_Images \
#     --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/results \
#     --save_name ${VERSION} \
#     --OCRBench_file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench.json \
#     --temperature 0 \
#     --conv_mode vicuna_v1 \
#     --num_workers 1

python src/model_vqa_ocrbench_qwen.py \
    --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_DRIP_10x_pretrain_NEW_DOWN_temp10_train_full \
    --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench_Images \
    --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/results \
    --save_name ${VERSION} \
    --OCRBench_file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench.json \
    --temperature 0 \
    --conv_mode qwen_v2 \
    --num_workers 1

# python src/model_vqa_ocrbench.py \
#     --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_FLASH_second_to_last_finetune_lora \
#     --model_base lmsys/vicuna-7b-v1.5 \
#     --image_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench_Images \
#     --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/results \
#     --save_name ${VERSION} \
#     --OCRBench_file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbench/OCRBench.json \
#     --temperature 0 \
#     --conv_mode vicuna_v1 \
#     --num_workers 1


conda deactivate
