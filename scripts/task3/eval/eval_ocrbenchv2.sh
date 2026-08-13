#!/bin/bash
#SBATCH --job-name=0812_OCRBenchV2_LLaVA_7B_SigLIP_HF_v2_DRIP_4x_temp15_new_downsample_train_full
#SBATCH --output=0812_OCRBenchV2_LLaVA_7B_SigLIP_HF_v2_DRIP_4x_temp15_new_downsample_train_full.log
#SBATCH --time=3:00:00
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

export MASTER_PORT=$((12000 + RANDOM % 20000))
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

cd /users/PAS2912/yusenpeng/DRIP/

VERSION="LLaVA_7B_SigLIP_HF_v2_DRIP_4x_temp15_new_downsample_train_full"


# python src/model_vqa_ocrbenchv2.py \
#     --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_finetune_train_full \
#     --dataset_path lmms-lab/OCRBench-v2 \
#     --dataset_split test \
#     --cache_dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2 \
#     --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/results \
#     --save_name ${VERSION} \
#     --num_workers 1 \
#     --temperature 0 \
#     --conv_mode vicuna_v1

python src/model_vqa_ocrbenchv2.py \
    --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_4x_temp15_new_downsample_train_full \
    --dataset_path lmms-lab/OCRBench-v2 \
    --dataset_split test \
    --cache_dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2 \
    --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/results \
    --save_name ${VERSION} \
    --num_workers 1 \
    --temperature 0 \
    --conv_mode vicuna_v1

# python src/model_vqa_ocrbenchv2_qwen.py \
#     --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_DRIP_4x_train_full \
#     --dataset_path lmms-lab/OCRBench-v2 \
#     --dataset_split test \
#     --cache_dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2 \
#     --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/results \
#     --save_name ${VERSION} \
#     --num_workers 1 \
#     --temperature 0 \
#     --conv_mode qwen_v2


# python src/model_vqa_ocrbenchv2.py \
#     --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_FLASH_second_to_last_finetune_lora \
#     --model_base lmsys/vicuna-7b-v1.5 \
#     --dataset_path lmms-lab/OCRBench-v2 \
#     --dataset_split test \
#     --cache_dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2 \
#     --output_folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/results \
#     --save_name ${VERSION} \
#     --num_workers 1 \
#     --temperature 0 \
#     --conv_mode vicuna_v1

conda deactivate


#### Evaluation ####
####################

conda activate OCRv2Eval


#### step1: merge json files

python src/OCRBenchv2_eval/json_merge.py \
    --pred /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/results/${VERSION}.json \
    --out /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/results/${VERSION}_merged.json

#### step2: run evaluation

export NLTK_DATA=/fs/scratch/PAS2836/yusenpeng_dataset/nltk_data
mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/scores

python src/OCRBenchv2_eval/eval.py \
    --input_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/results/${VERSION}_merged.json \
    --output_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/scores/${VERSION}_scores.json

python src/OCRBenchv2_eval/get_score.py \
    --json_file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/ocrbenchv2/scores/${VERSION}_scores.json

conda deactivate
