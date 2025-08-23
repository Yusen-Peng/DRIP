#!/bin/bash
#SBATCH --job-name=AUG_22_384_resolution_DRIP_4x_4_8
#SBATCH --output=AUG_22_384_resolution_DRIP_4x_4_8.txt
#SBATCH --time=168:00:00
#SBATCH --nodes=4
#SBATCH --ntasks=4                      
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP
source activate DRIP

export OMP_NUM_THREADS=16

cd /users/PAS2912/yusenpeng/Fast-CLIP/

# Get master node address (rank 0)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((12000 + RANDOM % 20000))

# Launch one torchrun per node (each launching 2 procs = 2 GPUs)

srun bash -c "
torchrun \
  --nproc_per_node=1 \
  --nnodes=4 \
  --node_rank=\$SLURM_PROCID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  src/task1_imagenet.py
"

# srun bash -c "
# torchrun --nproc_per_node=8 \
#     --nnodes=4 \
#     --node_rank=\$SLURM_PROCID \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#     src/task1_newcodebase.py\
#     --model vit_b_16 --epochs 300 --batch-size 512 --opt adamw --lr 0.003 --wd 0.3\
#     --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30\
#     --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra\
#     --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema
# "

# srun bash -c "
# torchrun --nproc_per_node=4 \
#     --nnodes=$SLURM_NNODES \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#     src/task1_newcodebase.py \
#       --model vit_b_16 --epochs 300 --batch-size 512 --opt adamw --lr 0.003 --wd 0.3 \
#       --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30 \
#       --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra \
#       --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema
# "

conda deactivate
