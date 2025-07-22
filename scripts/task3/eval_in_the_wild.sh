#!/bin/bash
#SBATCH --job-name=in_the_wild_eval
#SBATCH --output=in_the_wild_eval.txt
#SBATCH --time=00:20:00
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

python src/LLaVA_wrapper/llava_local/eval/model_vqa.py \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_in_the_wild/questions.jsonl \
    --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_in_the_wild/images \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_in_the_wild/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_in_the_wild/reviews
touch /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_in_the_wild/reviews/llava-v1.5-13b.jsonl

python src/LLaVA_wrapper/llava_local/eval/eval_gpt_review_bench.py \
    --question /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_in_the_wild/questions.jsonl \
    --context /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_in_the_wild/context.jsonl \
    --rule /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_in_the_wild/rule.json \
    --answer-list \
        /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_in_the_wild/answers_gpt4.jsonl \
        /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_in_the_wild/answers/llava-v1.5-13b.jsonl \
    --output \
        /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_in_the_wild/reviews/llava-v1.5-13b.jsonl

python src/LLaVA_wrapper/llava_local/eval/summarize_gpt_review.py -f /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_in_the_wild/reviews/llava-v1.5-13b.jsonl
