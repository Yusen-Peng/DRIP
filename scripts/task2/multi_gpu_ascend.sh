#!/bin/bash
#SBATCH --job-name=gradient_clipping_1e_3_DRIP_2X_32
#SBATCH --output=gradient_clipping_1e_3_DRIP_2X_32.log
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP
source activate DRIP

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))

cd /users/PAS2912/yusenpeng/Fast-CLIP/

torchrun --nproc_per_node=1 src/task2_clip.py 2>&1 | tee ablation.log

conda deactivate
# End of script