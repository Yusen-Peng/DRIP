import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2, DRIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    print("⚡️" * 20)
    print("Let's build the vision tower!")
    print("⚡️" * 20)

    # FIXME: all hardcoded values for now
    USE_DTP = True
    # FIXME: hardcoded checkpoint path, should be configurable
    checkpoint_path = "logs/2025_07_12-11_59_06-model_ViT-B-32-lr_0.0001-b_512-j_8-p_amp/checkpoints/epoch_1.pt"
    patch_size = 32
    compression_rate = 0.5
    lower_bound = False
    lambda_val = 1.0
    num_classes = 512
    delay_load = False
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        elif USE_DTP:
            return DRIPVisionTower(
                checkpoint_path=checkpoint_path, 
                vision_tower=vision_tower,
                args=vision_tower_cfg,
                patch_size=patch_size,
                compression_rate=compression_rate,
                lower_bound=lower_bound,
                lambda_val=lambda_val,
                num_classes=num_classes,
                delay_load=delay_load, 
                **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
