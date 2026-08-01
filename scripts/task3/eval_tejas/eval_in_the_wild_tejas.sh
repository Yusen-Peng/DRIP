#!/bin/bash
#SBATCH --job-name=Apr20_wild_fixed_8x
#SBATCH --output=Apr20_wild_fixed_8x.txt
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

cd /users/PAS3184/tejasnaik/DRIP/

# Load API key from .env
set -a
source /users/PAS3184/tejasnaik/DRIP/.env
set +a

VERSION="13b_ViT"

mkdir -p /fs/scratch/PAS2836/shared_LLaVA_eval/llava_bench_in_the_wild/answers
touch /fs/scratch/PAS2836/shared_LLaVA_eval/llava_bench_in_the_wild/answers/${VERSION}.jsonl

python src/model_vqa.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/llava-v1.5-13b-local \
    --question-file /fs/scratch/PAS2836/shared_LLaVA_eval/llava_bench_in_the_wild/questions.jsonl \
    --image-folder /fs/scratch/PAS2836/shared_LLaVA_eval/llava_bench_in_the_wild/images \
    --answers-file /fs/scratch/PAS2836/shared_LLaVA_eval/llava_bench_in_the_wild/answers/${VERSION}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /fs/scratch/PAS2836/shared_LLaVA_eval/llava_bench_in_the_wild/reviews
touch /fs/scratch/PAS2836/shared_LLaVA_eval/llava_bench_in_the_wild/reviews/${VERSION}.jsonl

python src/eval_gpt_review_bench.py \
    --question /fs/scratch/PAS2836/shared_LLaVA_eval/llava_bench_in_the_wild/questions.jsonl \
    --context /fs/scratch/PAS2836/shared_LLaVA_eval/llava_bench_in_the_wild/context.jsonl \
    --rule src/LLaVA_wrapper/llava_local/eval/table/rule.json \
    --answer-list \
        /fs/scratch/PAS2836/shared_LLaVA_eval/llava_bench_in_the_wild/answers_gpt4.jsonl \
        /fs/scratch/PAS2836/shared_LLaVA_eval/llava_bench_in_the_wild/answers/${VERSION}.jsonl \
    --output \
        /fs/scratch/PAS2836/shared_LLaVA_eval/llava_bench_in_the_wild/reviews/${VERSION}.jsonl

python src/summarize_gpt_review.py -f /fs/scratch/PAS2836/shared_LLaVA_eval/llava_bench_in_the_wild/reviews/${VERSION}.jsonl

conda deactivate
# End of script
