#!/bin/bash
#SBATCH --job-name=FLOP_measure
#SBATCH --output=FLOP_measure.txt
#SBATCH --time=00:08:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=PAS2836

module load miniconda3/24.1.2-py310
conda activate Fast-CLIP
source activate Fast-CLIP
cd /users/PAS2912/yusenpeng/Fast-CLIP/

# Run your training
python src/FLOP_measure.py

conda deactivate
# End of script