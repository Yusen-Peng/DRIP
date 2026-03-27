#!/bin/bash
#SBATCH --job-name=March26_DRIP_no_sample
#SBATCH --output=March26_DRIP_no_sample.txt
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP
source activate DRIP

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))

cd /users/PAS2912/yusenpeng/Fast-CLIP/

torchrun --nproc_per_node=1 src/task1_newcodebase.py \
    --model vit_b_16 --epochs 20 --batch-size 64 --opt adamw --lr 0.0003 --wd 0.3 \
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 2 \
    --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra \
    --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 \
    --output-dir /fs/scratch/PAS2836/yusenpeng_checkpoint/imagenet_DRIP_no_sample \
    --MODE DRIP --RATE 0.25 --TEMP 0.0

conda deactivate
# End of script