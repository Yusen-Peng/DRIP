#!/bin/bash
#SBATCH --job-name=0617_DocVQA_LLaVA_Qwen2.5-14B-Instruct_PruMerge_10x_train_full
#SBATCH --output=0617_DocVQA_LLaVA_Qwen2.5-14B-Instruct_PruMerge_10x_train_full.log
#SBATCH --time=00:40:00
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


VERSION="LLaVA_Qwen2.5-14B-Instruct_PruMerge_10x_train_full"
echo "Running LLaVA inference..."


# python src/model_vqa_loader.py \
#   --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_512_Fixed_8x_train_all \
#   --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/images \
#   --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/docvqa_validation_llava.jsonl \
#   --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/results/${VERSION}.jsonl \
#   --temperature 0 \
#   --conv-mode llava_v1


python src/model_vqa_loader_qwen.py \
  --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_train_full \
  --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/images \
  --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/docvqa_validation_llava.jsonl \
  --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/results/${VERSION}.jsonl \
  --temperature 0 \
  --conv-mode qwen_v2


# python src/model_vqa_loader.py \
#   --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_10x_second_to_last_train_lora \
#   --model-base lmsys/vicuna-7b-v1.5 \
#   --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/images \
#   --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/docvqa_validation_llava.jsonl \
#   --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/docvqa/results/${VERSION}.jsonl \
#   --temperature 0 \
#   --conv-mode llava_v1

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
