#!/bin/bash
#SBATCH --job-name=0826_DocVQA_Qwen3VL_SFT_DRIP_4x_10data_temp001_BP1e4
#SBATCH --output=0826_DocVQA_Qwen3VL_SFT_DRIP_4x_10data_temp001_BP1e4.log
#SBATCH --time=01:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
module load cuda/12.6.2
conda activate DRIP_qwenvl_flash

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune/qwenvl

VERSION="Qwen3VL_SFT_DRIP_4x_10data_temp001_BP1e4"
echo "Running LLaVA inference..."

# python eval_code/model_vqa_loader.py \
#   --model-path Qwen/Qwen3-VL-4B-Instruct \
#   --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/images \
#   --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/docvqa_validation_llava.jsonl \
#   --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/results/${VERSION}.jsonl \
#   --temperature 0

# python eval_code/model_vqa_loader.py \
#   --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_10 \
#   --model-base Qwen/Qwen3-VL-4B-Instruct \
#   --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/images \
#   --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/docvqa_validation_llava.jsonl \
#   --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/results/${VERSION}.jsonl \
#   --temperature 0

# python eval_code/model_vqa_loader.py \
#   --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_Fixed_4x_10data \
#   --model-base Qwen/Qwen3-VL-4B-Instruct \
#   --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/images \
#   --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/docvqa_validation_llava.jsonl \
#   --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/results/${VERSION}.jsonl \
#   --temperature 0 \
#   --merge-strategy Fixed \
#   --compression-rate 0.25


python eval_code/model_vqa_loader.py \
  --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_10data_temp001_BP1e4 \
  --model-base Qwen/Qwen3-VL-4B-Instruct \
  --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/images \
  --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/docvqa_validation_llava.jsonl \
  --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/results/${VERSION}.jsonl \
  --temperature 0 \
  --merge-strategy DRIP \
  --compression-rate 0.25 \
  --drip-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_DRIP_4x_10data_temp001_BP1e4/drip.bin


cd /users/PAS2912/yusenpeng/DRIP

echo "Evaluating ANLS..."
python src/DocVQA_eval/eval_docvqa_anls.py \
  --gt-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/docvqa_validation_gt.json \
  --pred-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/results/${VERSION}.jsonl \
  --out-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/results/${VERSION}_eval.json

echo "Aggregated score:"
python - <<EOF
import json
path = "/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/results/${VERSION}_eval.json"
with open(path) as f:
    data = json.load(f)
print(json.dumps(data["summary"], indent=2))
EOF

conda deactivate
