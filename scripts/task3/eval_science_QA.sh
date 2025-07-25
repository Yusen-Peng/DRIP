#!/bin/bash
#SBATCH --job-name=science_QA_eval
#SBATCH --output=science_QA_eval.txt
#SBATCH --time=00:30:00
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

mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/scienceqa/answers
touch /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/scienceqa/answers/llava-v1.5-13b.jsonl

python src/model_vqa_science.py \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/scienceqa/llava_test_QCM-LEA.json \
    --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/scienceqa/images/test \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

touch /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/scienceqa/answers/llava-v1.5-13b.jsonl
touch /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/scienceqa/answers/llava-v1.5-13b_output.jsonl
touch /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/scienceqa/answers/llava-v1.5-13b_result.json

python src/eval_science_qa.py \
    --base-dir /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/scienceqa \
    --result-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --output-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/scienceqa/answers/llava-v1.5-13b_output.jsonl \
    --output-result /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/scienceqa/answers/llava-v1.5-13b_result.json

conda deactivate
# End of script