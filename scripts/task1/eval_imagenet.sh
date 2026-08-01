#!/bin/bash
#SBATCH --job-name=May4_RE_eval_imagenet_DRIP
#SBATCH --output=May4_RE_eval_imagenet_DRIP.txt
#SBATCH --time=0:20:00
#SBATCH --ntasks=1
#SBATCH --partition=debug-nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP
source activate DRIP

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12001 + RANDOM % 20000))

cd /users/PAS2912/yusenpeng/DRIP/

torchrun --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    src/task1_newcodebase.py \
    --device cuda \
    --batch-size 64 \
    --workers 16 \
    --MODE DRIP --RATE 0.25 --TEMP 0.1 \
    --test-only \
    --resume /fs/scratch/PAS2836/yusenpeng_checkpoint/imagenet_DRIP_4x_01_warmup2/model_299.pth

conda deactivate
# End of script