#!/bin/bash
#SBATCH --job-name=0821_MMMU_LLaVA_7B_Fixed_4x_SCALE_train_full_2epochs_1500steps
#SBATCH --output=0821_MMMU_LLaVA_7B_Fixed_4x_SCALE_train_full_2epochs_1500steps.log
#SBATCH --time=00:45:00
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

cd /users/PAS2912/yusenpeng/DRIP/

VERSION="LLaVA_7B_Fixed_4x_SCALE_train_full_2epochs_1500steps"

python src/model_vqa_mmmu.py \
    --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Fixed_4x_SCALE_train_full_2epochs/checkpoint-1500 \
    --output_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmmu/answers/${VERSION}.json \
    --config_path src/LLaVA_wrapper/llava_local/mmmu_utils/llava.yaml

# python src/model_vqa_mmmu.py \
#     --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_4x_temp001_new_downsample_train_full \
#     --output_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmmu/answers/${VERSION}.json \
#     --config_path src/LLaVA_wrapper/llava_local/mmmu_utils/llava.yaml

# python src/model_vqa_mmmu_qwen.py \
#     --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_DRIP_4x_train_full \
#     --output_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmmu/answers/${VERSION}.json \
#     --config_path src/LLaVA_wrapper/llava_local/mmmu_utils/llava.yaml

# python src/model_vqa_mmmu.py \
#     --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_FLASH_second_to_last_finetune_lora \
#     --model_base lmsys/vicuna-7b-v1.5 \
#     --output_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmmu/answers/${VERSION}.json \
#     --config_path src/LLaVA_wrapper/llava_local/mmmu_utils/llava.yaml

python src/mmmu_main_eval_only.py \
    --output_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmmu/answers/${VERSION}.json \
    --answer_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmmu/answer_key/answer_dict_val.json
