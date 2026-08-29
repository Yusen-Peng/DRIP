#!/bin/bash
#SBATCH --job-name=0826_textVQA_Qwen3VL_SFT_DRIP_4x_10data_temp001
#SBATCH --output=0826_textVQA_Qwen3VL_SFT_DRIP_4x_10data_temp001.log
#SBATCH --time=00:55:00
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
export CUDA_LAUNCH_BLOCKING=1

VERSION="Qwen3VL_SFT_DRIP_4x_10data_temp001"

cd /users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune/qwenvl

# python eval_code/model_vqa_loader.py \
#     --model-path Qwen/Qwen3-VL-4B-Instruct \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/train_images \
#     --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/answers/${VERSION}.jsonl \
#     --temperature 0

# python eval_code/model_vqa_loader.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_10 \
#     --model-base Qwen/Qwen3-VL-4B-Instruct \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/train_images \
#     --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/answers/${VERSION}.jsonl \
#     --temperature 0


# python eval_code/model_vqa_loader.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_Fixed_4x_10data \
#     --model-base Qwen/Qwen3-VL-4B-Instruct \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/train_images \
#     --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/answers/${VERSION}.jsonl \
#     --temperature 0 \
#     --merge-strategy Fixed \
#     --compression-rate 0.25

python eval_code/model_vqa_loader.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_10data_temp001 \
    --model-base Qwen/Qwen3-VL-4B-Instruct \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/train_images \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/answers/${VERSION}.jsonl \
    --temperature 0 \
    --merge-strategy DRIP \
    --compression-rate 0.25 \
    --drip-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_10data_temp001/drip.bin

cd /users/PAS2912/yusenpeng/DRIP

python src/eval_textvqa.py \
    --annotation-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/TextVQA_0.5.1_val.json \
    --result-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/answers/${VERSION}.jsonl
