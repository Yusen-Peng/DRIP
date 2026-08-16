import sys
import torch

"""
How to run this script:

salloc --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A PAS2836 --partition debug-nextgen --time 00:30:00
module load miniconda3/24.1.2-py310
conda activate DRIP
python src/example_analysis/MAE_recon.py
"""

PROJECT_ROOT = "/users/PAS2912/yusenpeng/DRIP"
sys.path.insert(0, PROJECT_ROOT)

import src.example_analysis.mae_utils.models_mae as models_mae

model = models_mae.mae_vit_large_patch16()

checkpoint = torch.load("/users/PAS2912/yusenpeng/mae_pretrain_vit_large_full.pth", map_location="cpu")

model.load_state_dict(checkpoint["model"], strict=True)

model = model.cuda()
model.eval()

print("Loaded pretrained MAE successfully")