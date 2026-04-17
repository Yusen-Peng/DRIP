#!/bin/bash
#SBATCH --job-name=flash-attn-test
#SBATCH --account=PAS2836
#SBATCH --partition=debug-nextgen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=flash-attn-test.out
#SBATCH --error=flash-attn-test.err

export MAX_JOBS=4
export NVCC_THREADS=2
export TORCH_CUDA_ARCH_LIST="9.0"

module load miniconda3/24.1.2-py310
conda deactivate
conda activate DRIP_flash

python - <<'PY'
import torch
from flash_attn import flash_attn_func

q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.float16)
k = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.float16)
v = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.float16)

out = flash_attn_func(q, k, v)
print(out.shape, out.dtype, out.device)
PY

conda deactivate