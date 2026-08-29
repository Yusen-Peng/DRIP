#!/bin/bash
#SBATCH --job-name=June3_ChartQAPro_LLaVA_7B_DRIP_10x_second_to_last_train_lora
#SBATCH --output=June3_ChartQAPro_LLaVA_7B_DRIP_10x_second_to_last_train_lora.log
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
module load cuda/12.6.2
conda activate DRIP

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /users/PAS3184/tejasnaik/DRIP


VERSION="LLaVA_13B_ViT"
echo "Running LLaVA inference..."


# python src/model_vqa_chartqapro.py \
#   --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_FLASH_second_to_last_finetune_full_TRAIN_VIT \
#   --image-folder /fs/scratch/PAS2836/shared_LLaVA_eval/chartvqapro/images \
#   --question-file /fs/scratch/PAS2836/shared_LLaVA_eval/chartvqapro/chartqapro_test_llava.jsonl \
#   --answers-file /fs/scratch/PAS2836/shared_LLaVA_eval/chartvqapro/results/${VERSION}.jsonl \
#   --temperature 0 \
#   --conv-mode llava_v1

python src/model_vqa_chartqapro.py \
  --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_FLASH_second_to_last_finetune_full_TRAIN_VIT \
  --image-folder /fs/scratch/PAS2836/shared_LLaVA_eval/chartvqapro/images \
  --question-file /fs/scratch/PAS2836/shared_LLaVA_eval/chartvqapro/chartqapro_test_llava.jsonl \
  --answers-file /fs/scratch/PAS2836/shared_LLaVA_eval/chartvqapro/results/${VERSION}.jsonl \
  --temperature 0 \
  --conv-mode llava_v1


python src/ChartQAPro_eval/eval_prediction.py \
  --predictions-file /fs/scratch/PAS2836/shared_LLaVA_eval/chartvqapro/results/${VERSION}.jsonl \
  --gt-file /fs/scratch/PAS2836/shared_LLaVA_eval/chartvqapro/chartqapro_test_gt.json

conda deactivate
