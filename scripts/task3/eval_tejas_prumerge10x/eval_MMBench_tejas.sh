#!/bin/bash
#SBATCH --job-name=Apr15_MMBench_fixed_8x
#SBATCH --output=Apr15_MMBench_fixed_8x.txt
#SBATCH --time=00:25:00
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

VERSION="13b_ViT"

mkdir -p /fs/scratch/PAS2836/shared_LLaVA_eval/MMBench/answers
mkdir -p /fs/scratch/PAS2836/shared_LLaVA_eval/MMBench/answers/mmbench_dev_20230712
touch /fs/scratch/PAS2836/shared_LLaVA_eval/MMBench/answers/mmbench_dev_20230712/${VERSION}.jsonl

python src/model_vqa_mmbench.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/llava-v1.5-7b-local \
    --question-file /fs/scratch/PAS2836/shared_LLaVA_eval/MMBench/mmbench_dev_20230712.tsv \
    --answers-file /fs/scratch/PAS2836/shared_LLaVA_eval/MMBench/answers/mmbench_dev_20230712/${VERSION}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /fs/scratch/PAS2836/shared_LLaVA_eval/MMBench/answers_upload/mmbench_dev_20230712

python src/convert_mmbench_for_submission.py \
    --annotation-file /fs/scratch/PAS2836/shared_LLaVA_eval/MMBench/mmbench_dev_20230712.tsv \
    --result-dir /fs/scratch/PAS2836/shared_LLaVA_eval/MMBench/answers/mmbench_dev_20230712 \
    --upload-dir /fs/scratch/PAS2836/shared_LLaVA_eval/MMBench/answers_upload/mmbench_dev_20230712 \
    --experiment ${VERSION}

echo "Evaluation completed."