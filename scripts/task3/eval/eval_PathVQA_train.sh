#!/bin/bash
#SBATCH --job-name=PathVQA_infer_LLaVA_7B_PruMerge_10x_train
#SBATCH --output=PathVQA_infer_LLaVA_7B_PruMerge_10x_train.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
module load cuda/12.6.2
conda activate DRIP_flash

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /users/PAS2912/yusenpeng/DRIP


VERSION="LLaVA_7B_PruMerge_10x"

MODEL_PATH="/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_FLASH_finetune_ALL_ONCE_full"

DATA_ROOT="/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/PathVQA"
ANSWER_ROOT="${DATA_ROOT}/answers"
mkdir -p "${ANSWER_ROOT}"
echo "========================================"
echo "Running PathVQA training"
echo "========================================"
python src/model_vqa_loader_pathvqa.py \
    --model-path "${MODEL_PATH}" \
    --question-file "${DATA_ROOT}/train.json" \
    --image-folder "${DATA_ROOT}/images/train" \
    --answers-file "${ANSWER_ROOT}/train_${VERSION}.jsonl" \
    --temperature 0 \
    --max_new_tokens 32

python src/eval_pathvqa.py \
    --ground-truth /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/PathVQA/train.json \
    --predictions "/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/PathVQA/answers/train_${VERSION}.jsonl"
