#!/bin/bash
#SBATCH --job-name=BioCLIP_train_DRIP_4x
#SBATCH --output=BioCLIP_train_DRIP_4x.txt
#SBATCH --time=168:00:00
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

# --- Memory stability + DDP fail-fast ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1   # <- new name replaces NCCL_ASYNC_ERROR_HANDLING
export NCCL_DEBUG=WARN
export CUDNN_BENCHMARK=0                   # optional but recommended

torchrun --nproc_per_node 4 -m src.training.main \
  --train-data '/fs/scratch/PAS2836/yusenpeng_dataset/bioclip/data/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/train/shard-{000000..000164}.tar' \
  --val-data '/fs/scratch/PAS2836/yusenpeng_dataset/bioclip/data/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/val/shard-{000000..000034}.tar' \
  --dataset-type 'webdataset' \
  --train-num-samples 10_000_000 \
  --val-num-samples 500_000 \
  --pretrained '/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_4x_16_ViT_4_8/checkpoints/epoch_15.pt' \
  --text_type 'random' \
  --warmup 1000 \
  --batch-size 512 \
  --accum-freq 2 \
  --epochs 30 \
  --workers 8 \
  --model ViT-B-16 \
  --lr 1e-4 \
  --log-every-n-steps 1 \
  --dataset-resampled \
