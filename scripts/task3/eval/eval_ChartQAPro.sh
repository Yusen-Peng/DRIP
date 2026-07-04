#!/bin/bash
#SBATCH --job-name=0702_ChartQAPro_LLaVA_7B_Perceiver_10x_train_all
#SBATCH --output=0702_ChartQAPro_LLaVA_7B_Perceiver_10x_train_all.log
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=quad
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


VERSION="LLaVA_7B_Perceiver_10x_train_all"
echo "Running LLaVA inference..."


python src/model_vqa_chartqapro.py \
  --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Perceiver_10x_train_all \
  --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/images \
  --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/chartqapro_test_llava.jsonl \
  --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/results/${VERSION}.jsonl \
  --temperature 0 \
  --conv-mode llava_v1

# python src/model_vqa_chartqapro_qwen.py \
#   --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_DRIP_10x_train_full \
#   --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/images \
#   --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/chartqapro_test_llava.jsonl \
#   --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/results/${VERSION}.jsonl \
#   --temperature 0 \
#   --conv-mode qwen_v2



# python src/model_vqa_chartqapro.py \
#   --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_10x_second_to_last_train_lora \
#   --model-base lmsys/vicuna-7b-v1.5 \
#   --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/images \
#   --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/chartqapro_test_llava.jsonl \
#   --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/results/${VERSION}.jsonl \
#   --temperature 0 \
#   --conv-mode llava_v1


python src/ChartQAPro_eval/eval_prediction.py \
  --predictions-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/results/${VERSION}.jsonl \
  --gt-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/chartvqapro/chartqapro_test_gt.json

conda deactivate
