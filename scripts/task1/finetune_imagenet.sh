#!/bin/bash
#SBATCH --job-name=AUG_23_ViT_recheck
#SBATCH --output=AUG_23_ViT_recheck.txt
#SBATCH --time=80:00:00
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

cd /users/PAS2912/yusenpeng/Fast-CLIP/

torchrun --nproc_per_node=4 src/task1_newcodebase.py \
    --model vit_b_16 --epochs 300 --batch-size 256 --opt adamw --lr 0.003 --wd 0.3 \
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30 \
    --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra \
    --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema

conda deactivate
# End of script