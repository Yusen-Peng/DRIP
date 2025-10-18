#!/bin/bash
#SBATCH --job-name=VIT_ZERO_SHOT
#SBATCH --output=VIT_ZERO_SHOT.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1                      
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP
source activate DRIP

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))

cd /users/PAS2912/yusenpeng/Fast-CLIP/bioclip
export PYTHONPATH="$PWD:$PYTHONPATH"


#export DATA_ROOT="/fs/scratch/PAS2836/yusenpeng_dataset/bioclip/data/eval/PLK_Mini"
#export DATA_ROOT="/fs/scratch/PAS2836/yusenpeng_dataset/bioclip/data/eval/INS_Mini"
#export DATA_ROOT="/fs/scratch/PAS2836/yusenpeng_dataset/bioclip/data/eval/INS_2_Mini"
#export DATA_ROOT="/fs/scratch/PAS2836/yusenpeng_dataset/bioclip/data/eval/PLT_NET_Mini"
#export DATA_ROOT="/fs/scratch/PAS2836/yusenpeng_dataset/bioclip/data/eval/FNG_Mini"
#export DATA_ROOT="/fs/scratch/PAS2836/yusenpeng_dataset/bioclip/data/eval/PLT_VIL_Mini"
#export DATA_ROOT="/fs/scratch/PAS2836/yusenpeng_dataset/bioclip/data/eval/MED_LF_Mini"
#export DATA_ROOT="/fs/scratch/PAS2836/yusenpeng_dataset/bioclip/data/eval/PLT_DOC_Mini"
export DATA_ROOT="/fs/scratch/PAS2836/yusenpeng_dataset/bioclip/data/eval/birds_525"


#export PRETRAINED="openai"
#export PRETRAINED="/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/ViT_B_16/checkpoints/epoch_15.pt"
#export PRETRAINED="/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_4x_16_ViT_4_8/checkpoints/epoch_15.pt"
#export PRETRAINED="/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_10x_16_ViT_4_8/checkpoints/epoch_15.pt"
export PRETRAINED="/fs/scratch/PAS2836/yusenpeng_checkpoint/BioCLIP/ViT/checkpoints/epoch_30.pt"

export TEXT_TYPE="asis"
#export TEXT_TYPE="taxon_com"
#"com", "sci", "sci_com", "taxon_com", "asis"
export LABEL_FILE="metadata.csv"
export LOG_FILEPATH="../storage/logs"

python -m src.evaluation.zero_shot_iid \
        --model "ViT-B-16" \
        --batch-size 256 \
        --data_root $DATA_ROOT \
        --pretrained $PRETRAINED \
        --label_filename $LABEL_FILE \
        --log $LOG_FILEPATH \
        --text_type $TEXT_TYPE \
