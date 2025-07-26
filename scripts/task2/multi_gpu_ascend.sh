#!/bin/bash
#SBATCH --job-name=DRIP-2x-32-11-resume
#SBATCH --output=DRIP-2x-32-11-resume.log
#SBATCH --time=70:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate DRIP
source activate DRIP

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))

cd /users/PAS2912/yusenpeng/Fast-CLIP/

#python src/train_CLIP.py
#torchrun --nproc_per_node=4 src/train_CLIP.py
torchrun --nproc_per_node=4 src/task2_clip.py 2>&1 | tee ablation.log

conda deactivate
# End of script