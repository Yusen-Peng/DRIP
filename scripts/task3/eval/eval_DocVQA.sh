#!/bin/bash

#SBATCH --job-name=docvqa-llava-eval

#SBATCH --account=PAS2836

#SBATCH --partition=nextgen

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8

#SBATCH --mem=96G

#SBATCH --time=06:00:00

#SBATCH --output=docvqa_llava_eval.out

#SBATCH --error=docvqa_llava_eval.err

module load miniconda3/24.1.2-py310

module load cuda/12.6.2

conda activate DRIP_flash

export OMP_NUM_THREADS=8

export TOKENIZERS_PARALLELISM=false

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /users/PAS2912/yusenpeng/DRIP

MODEL_PATH="/path/to/your/llava/checkpoint"

WORK_DIR="/fs/scratch/PAS2836/yusenpeng_docvqa"

SPLIT="validation"

mkdir -p "${WORK_DIR}"

echo "Preparing DocVQA..."

python prepare_docvqa_llava.py \

  --dataset lmms-lab/DocVQA \

  --split ${SPLIT} \

  --out-dir "${WORK_DIR}" \

  --prompt-style short

echo "Running LLaVA inference..."

python model_vqa_loader.py \

  --model-path "${MODEL_PATH}" \

  --model-base None \

  --image-folder "${WORK_DIR}/images" \

  --question-file "${WORK_DIR}/docvqa_${SPLIT}_llava.jsonl" \

  --answers-file "${WORK_DIR}/answers_docvqa_${SPLIT}.jsonl" \

  --conv-mode llava_v1 \

  --temperature 0 \

  --num-beams 1 \

  --max-new-tokens 32

echo "Evaluating ANLS..."

python eval_docvqa_anls.py \

  --gt-file "${WORK_DIR}/docvqa_${SPLIT}_gt.json" \

  --pred-file "${WORK_DIR}/answers_docvqa_${SPLIT}.jsonl" \

  --out-file "${WORK_DIR}/docvqa_${SPLIT}_eval.json"

echo "Done. Results:"

cat "${WORK_DIR}/docvqa_${SPLIT}_eval.json" | head -40