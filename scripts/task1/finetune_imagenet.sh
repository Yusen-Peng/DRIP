#!/bin/bash
#SBATCH --job-name=Apr1_DRIP_4x_01temp_warmup5
#SBATCH --output=Apr1_DRIP_4x_01temp_warmup5.txt
#SBATCH --time=80:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP
source activate DRIP

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12001 + RANDOM % 20000))

cd /users/PAS2912/yusenpeng/Fast-CLIP/

torchrun --nproc_per_node=4 src/task1_newcodebase.py \
    --model vit_b_16 --epochs 300 --batch-size 64 --opt adamw --lr 0.0003 --wd 0.3 \
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 5 \
    --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra \
    --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 \
    --output-dir /fs/scratch/PAS2836/yusenpeng_checkpoint/imagenet_DRIP_4x_01_warmup5 \
    --MODE DRIP --RATE 0.25 --TEMP 0.1

conda deactivate
# End of script