#!/bin/bash
#SBATCH --job-name=0821_MMVet_NEW_DOWN_10x_to_8x
#SBATCH --output=0821_MMVet_NEW_DOWN_10x_to_8x.log
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP_flash
source activate DRIP_flash

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))

VERSION="NEW_DOWN_10x_to_8x"

cd /users/PAS2912/yusenpeng/DRIP/

python src/model_vqa_loader.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_10x_pretrain_NEW_DOWN_temp10_train_full \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/images \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/answers/${VERSION}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# python src/model_vqa_loader.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_4x_temp001_new_downsample_train_full \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/llava-mm-vet.jsonl \
#     --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/images \
#     --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/answers/${VERSION}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python src/model_vqa_loader_qwen.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_DRIP_4x_pretrain_NEW_DOWN_temp001_train_full \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/llava-mm-vet.jsonl \
#     --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/images \
#     --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/answers/${VERSION}.jsonl \
#     --temperature 0 \
#     --conv-mode qwen_v2

# python src/model_vqa_loader.py \
#     --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_FLASH_second_to_last_finetune_lora \
#     --model-base lmsys/vicuna-7b-v1.5 \
#     --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/llava-mm-vet.jsonl \
#     --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/images \
#     --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/answers/${VERSION}.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

mkdir -p /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/results
python src/convert_mmvet_for_eval.py \
    --src /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/answers/${VERSION}.jsonl \
    --dst /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mm-vet/results/${VERSION}.json

echo "Evaluation completed for ${VERSION}. Please submit the result json file to: https://huggingface.co/spaces/whyu/MM-Vet_Evaluator"
conda deactivate
# End of script
