#!/bin/bash
#SBATCH --job-name=Qwen3VL_SFT_10
#SBATCH --output=logs/Qwen3VL_SFT_10.out
#SBATCH --account=PAS2836
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=22:00:00

module load miniconda3/24.1.2-py310
conda deactivate
conda activate DRIP_qwenvl_flash
cd /users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune
mkdir -p logs

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 20001-29999 -n 1)

export NPROC_PER_NODE=1
export WORLD_SIZE=1


export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4


bash scripts/sft_qwen3_4b.sh