#!/bin/bash
#SBATCH --job-name=Feb8_CLIP_fixed_pooling_4x_pitzer
#SBATCH --output=Feb8_CLIP_fixed_pooling_4x_pitzer.log
#SBATCH --time=95:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP
source activate DRIP

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))

cd /users/PAS2912/yusenpeng/Fast-CLIP/

torchrun --nproc_per_node=4 src/task2_clip.py

conda deactivate
# End of script