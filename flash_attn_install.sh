#!/bin/bash
#SBATCH --job-name=flash-attn-build
#SBATCH --account=PAS2836
#SBATCH --partition=debug-nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=flash-attn-build.out
#SBATCH --error=flash-attn-build.err
export MAX_JOBS=4
export NVCC_THREADS=2
export TORCH_CUDA_ARCH_LIST="9.0"
export TMPDIR=/users/PAS3184/tejasnaik/tmp
mkdir -p /users/PAS3184/tejasnaik/tmp
module load miniconda3/24.1.2-py310
module load cuda/12.6.2
conda activate DRIP_flash
python -m pip install -v flash-attn --no-build-isolation
conda deactivate
