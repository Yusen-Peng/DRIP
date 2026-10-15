#!/bin/bash
#SBATCH --job-name=MMERealWorld_LLaVA_7B_original
#SBATCH --output=MMERealWorld_LLaVA_7B_original.log
#SBATCH --time=01:20:00
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

VERSION="LLaVA_7B_original"

python src/model_vqa_mme_real_world.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_FLASH_finetune_ALL_ONCE_full \
    --question-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMErealworld/MME-RealWorld.json \
    --image-folder /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMErealworld/images \
    --answers-file /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMErealworld/answers/${VERSION}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1


cd /fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/MMErealworld

python eval_MME_realworld.py \
    --results_file answers/${VERSION}.jsonl


conda deactivate
# End of script