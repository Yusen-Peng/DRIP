#!/bin/bash
#SBATCH --job-name=Birds_FEW_SHOT
#SBATCH --output=Birds_FEW_SHOT.txt
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP
# source activate DRIP         # (redundant)
export OMP_NUM_THREADS=8
export MASTER_PORT=$((12000 + RANDOM % 20000))

# Keep dataloader light
export BIOCLIP_WORKERS=4        # if script reads this; see code patch below
export BIOCLIP_PREFETCH=2
export BIOCLIP_PINMEM=0

# Chunking control for logits (used in the code patch below)
export BIOCLIP_LOGIT_CHUNK=8192

cd /users/PAS2912/yusenpeng/Fast-CLIP/bioclip
export PYTHONPATH="$PWD:$PYTHONPATH"

export DATA_ROOT="/fs/scratch/PAS2836/yusenpeng_dataset/bioclip/data/eval/birds_525"


#export PRETRAINED="openai"
#export PRETRAINED="/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/ViT_B_16/checkpoints/epoch_15.pt"
#export PRETRAINED="/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_4x_16_ViT_4_8/checkpoints/epoch_15.pt"
export PRETRAINED="/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_10x_16_ViT_4_8/checkpoints/epoch_15.pt"




export TASK_TYPE="all"
export LABEL_FILE="metadata.csv"
export LOG_FILEPATH="../storage/logs"

python -m src.evaluation.few_shot \
      --model "ViT-B-16" \
      --batch-size 32 \
      --data_root $DATA_ROOT \
      --pretrained $PRETRAINED \
      --label_filename $LABEL_FILE \
      --log $LOG_FILEPATH \
      --task_type $TASK_TYPE \
      --nfold 5 \
      --kshot_list 1 \