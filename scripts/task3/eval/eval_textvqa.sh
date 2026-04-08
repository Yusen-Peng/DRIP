#!/bin/bash
#SBATCH --job-name=textvqa_DRIP_XL
#SBATCH --output=textvqa_DRIP_XL.txt
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

python src/model_vqa_loader.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/XLbased-DRIP-10x-16-5-7-finetune \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textvqa/train_images \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textvqa/answers/DRIP.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python src/eval_textvqa.py \
    --annotation-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textvqa/answers/DRIP.jsonl
