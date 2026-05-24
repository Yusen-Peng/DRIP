#!/bin/bash
#SBATCH --job-name=May23_VisWiz_TEST
#SBATCH --output=May23_VisWiz_TEST.log
#SBATCH --time=00:40:00
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

VERSION="VisWiz_TEST"

cd /users/PAS2912/yusenpeng/DRIP/
mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VisWiz/answers
touch /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VisWiz/answers/${VERSION}.jsonl

python src/model_vqa_loader.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/llava-v1.5-7b-local \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VizWiz/llava_test.jsonl \
    --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VizWiz/test \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VizWiz/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python src/convert_vizwiz_for_submission.py \
    --annotation-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VizWiz/llava_test.jsonl \
    --result-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VizWiz/answers/llava-v1.5-7b.jsonl \
    --result-upload-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VizWiz/answers_upload/llava-v1.5-7b.json