#!/bin/bash
#SBATCH --job-name=May25_wild_LLaVA_7B_Fixed_100x_finetune_train_full
#SBATCH --output=May25_wild_LLaVA_7B_Fixed_100x_finetune_train_full.log
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

cd /users/PAS2912/yusenpeng/DRIP/

# Load API key from .env
set -a
source /users/PAS2912/yusenpeng/DRIP/.env
set +a

VERSION="LLaVA_7B_Fixed_100x_finetune_train_full"

mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_bench_in_the_wild/answers
touch /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_bench_in_the_wild/answers/${VERSION}.jsonl

python src/model_vqa.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Fixed_100x_finetune_train_full \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_bench_in_the_wild/questions.jsonl \
    --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_bench_in_the_wild/images \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_bench_in_the_wild/answers/${VERSION}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# python src/model_vqa.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Fixed_100x_finetune_train_lora \
#     --model-base lmsys/vicuna-7b-v1.5 \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_bench_in_the_wild/questions.jsonl \
#     --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_bench_in_the_wild/images \
#     --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_bench_in_the_wild/answers/${VERSION}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_bench_in_the_wild/reviews
touch /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_bench_in_the_wild/reviews/${VERSION}.jsonl

python src/eval_gpt_review_bench.py \
    --question /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_bench_in_the_wild/questions.jsonl \
    --context /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_bench_in_the_wild/context.jsonl \
    --rule src/LLaVA_wrapper/llava_local/eval/table/rule.json \
    --answer-list \
        /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_bench_in_the_wild/answers_gpt4.jsonl \
        /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_bench_in_the_wild/answers/${VERSION}.jsonl \
    --output \
        /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_bench_in_the_wild/reviews/${VERSION}.jsonl

python src/summarize_gpt_review.py -f /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/llava_bench_in_the_wild/reviews/${VERSION}.jsonl

conda deactivate
# End of script
