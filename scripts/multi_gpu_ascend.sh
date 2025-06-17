#!/bin/bash
#SBATCH --job-name=COCO_CLIP
#SBATCH --output=COCO_CLIP.txt
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate Fast-CLIP
source activate Fast-CLIP

export OMP_NUM_THREADS=16
export MASTER_PORT=$((12000 + RANDOM % 20000))

cd /users/PAS2912/yusenpeng/Fast-CLIP/

#python src/train_CLIP.py
#torchrun --nproc_per_node=4 src/train_CLIP.py
torchrun --nproc_per_node=4 src/train_CLIP.py 2>&1 | tee full.log

conda deactivate
# End of script