import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2, DRIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):

    # FIXME: all hardcoded. Need to be fixed later.
    USE_DTP = True 
    FINETUNING_MODE = False
    if USE_DTP:
        print("🍟" * 20)
        print("Using DTP-ViT as the vision tower")
        print("🍟" * 20)
    checkpoint_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/ImageNet_DRIP_78/model_299.pth"
    patch_size = 16
    compression_rate = 0.5
    depth = (4, 8, 0)

    lower_bound = False
    lambda_val = 1.0
    num_classes = 512


    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        elif USE_DTP:
            print("🍟" * 20)
            print(f"Using DTP-ViT from the path {checkpoint_path}")
            print("🍟" * 20)
            return DRIPVisionTower(
                checkpoint_path=checkpoint_path, 
                vision_tower=vision_tower,
                args=vision_tower_cfg,
                patch_size=patch_size,
                compression_rate=compression_rate,
                lower_bound=lower_bound,
                lambda_val=lambda_val,
                depth=depth,
                num_classes=num_classes,
                finetuning_mode=FINETUNING_MODE,
                **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
