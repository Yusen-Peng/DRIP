#!/bin/bash
#SBATCH --job-name=Apr9_textVQA_fixed_10x
#SBATCH --output=Apr9_textVQA_fixed_10x.txt
#SBATCH --time=00:30:00
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

python src/model_vqa_loader.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/llava-v1.5-7b-local \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/train_images \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/answers/fixed_10x.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python src/eval_textvqa.py \
    --annotation-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/TextVQA_0.5.1_val.json \
    --result-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/textVQA/answers/fixed_10x.jsonl
