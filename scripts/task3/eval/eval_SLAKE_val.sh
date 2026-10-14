#!/bin/bash
#SBATCH --job-name=SLAKE_infer_LLaVA_7B_PruMerge_10x_validate
#SBATCH --output=SLAKE_infer_LLaVA_7B_PruMerge_10x_validate.log
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

DATA_ROOT="/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/SLAKE"
ANSWER_ROOT="${DATA_ROOT}/answers"

mkdir -p "${ANSWER_ROOT}"


echo "========================================"
echo "Running SLAKE validate inference"
echo "Model: ${VERSION}"
echo "========================================"

python src/model_vqa_loader_slake.py \
    --model-path "${MODEL_PATH}" \
    --question-file "${DATA_ROOT}/validate.json" \
    --image-folder "${DATA_ROOT}/imgs" \
    --answers-file "${ANSWER_ROOT}/validate_${VERSION}.jsonl" \
    --temperature 0 \
    --max_new_tokens 32


echo "========================================"
echo "Evaluating SLAKE validate"
echo "========================================"

python src/eval_slake.py \
    --ground-truth "${DATA_ROOT}/validate.json" \
    --predictions "${ANSWER_ROOT}/validate_${VERSION}.jsonl"