import sys
import torch
sys.path.insert(0, "/users/PAS2912/yusenpeng/mae")
import models_mae

model = models_mae.mae_vit_large_patch16()

checkpoint = torch.load(
    "../mae/mae_pretrain_vit_large.pth",
    map_location="cpu",
)

model.load_state_dict(checkpoint["model"], strict=True)

model = model.cuda()
model.eval()

print("Loaded pretrained MAE successfully")