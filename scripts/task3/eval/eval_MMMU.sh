#!/bin/bash
#SBATCH --job-name=May24_MMMU_TEST_lora
#SBATCH --output=May24_MMMU_TEST_lora.log
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP
source activate DRIP

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))

cd /users/PAS2912/yusenpeng/DRIP/

VERSION="LLaVA_7B_checkpoint_lora"

# python src/model_vqa_mmmu.py \
#     --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/llava-v1.5-7b-local \
#     --output_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmmu/answers/${VERSION}.json \
#     --config_path src/LLaVA_wrapper/llava_local/mmmu_utils/llava.yaml

python src/model_vqa_mmmu.py \
    --model_path /fs/scratch/PAS2836/yusenpeng_checkpoint/llava-v1.5-7b-lora-local \
    --model_base lmsys/vicuna-7b-v1.5 \
    --output_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmmu/answers/${VERSION}.json \
    --config_path src/LLaVA_wrapper/llava_local/mmmu_utils/llava.yaml


python src/mmmu_main_eval_only.py \
    --output_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmmu/answers/${VERSION}.json \
    --answer_path /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/mmmu/answer_key/answer_dict_val.json
