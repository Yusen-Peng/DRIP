#!/bin/bash
#SBATCH --job-name=0826_ChartQAPro_Qwen3VL_Fixed_4x_checkpoint_12
#SBATCH --output=0826_ChartQAPro_Qwen3VL_Fixed_4x_checkpoint_12.log
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
conda activate DRIP_qwenvl_flash

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune/qwenvl

VERSION="0826_ChartQAPro_Qwen3VL_Fixed_4x_checkpoint_12"
echo "Running LLaVA inference..."


# python eval_code/model_vqa_chartqapro.py \
#   --model-path Qwen/Qwen3-VL-4B-Instruct \
#   --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/images \
#   --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/chartqapro_test_llava.jsonl \
#   --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/results/${VERSION}.jsonl \
#   --temperature 0 \
#   --conv-mode llava_v1


# python eval_code/model_vqa_chartqapro.py \
#   --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_10 \
#   --model-base Qwen/Qwen3-VL-4B-Instruct \
#   --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/images \
#   --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/chartqapro_test_llava.jsonl \
#   --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/results/${VERSION}.jsonl \
#   --temperature 0 \
#   --conv-mode llava_v1


python eval_code/model_vqa_chartqapro.py \
  --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_Fixed_DEBUG/checkpoint-12 \
  --model-base Qwen/Qwen3-VL-4B-Instruct \
  --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/images \
  --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/chartqapro_test_llava.jsonl \
  --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/results/${VERSION}.jsonl \
  --temperature 0 \
  --conv-mode llava_v1 \
  --merge-strategy Fixed \
  --compression-rate 0.25




cd /users/PAS2912/yusenpeng/DRIP

python src/ChartQAPro_eval/eval_prediction.py \
  --predictions-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/results/${VERSION}.jsonl \
  --gt-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/chartqapro_test_gt.json

conda deactivate