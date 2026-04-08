import os
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from .clip_encoder import CLIPVisionTowerS2, DRIPVisionTower, ViTVisionTower, BaselineVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):

    # FIXME: all hardcoded. Need to be fixed later.
    MODE = "ViT" # "ViT" or "DRIP" "Fixed"
    FINETUNING_MODE = True
    # depth = (4, 8, 0)
    depth = 12
    patch_size = 16
    compression_rate = 0.25


    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    print(f"🍌🍌🍌 Vision tower: {vision_tower}, exists: {is_absolute_path_exists} 🍌🍌🍌")
    checkpoint_path = vision_tower
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        elif MODE == "DRIP":
            print("🍟" * 20)
            print(f"Using DTP-ViT from the path {checkpoint_path}")
            print("🍟" * 20)
            # return DRIPVisionTower(
            #     backbone=BACKBONE,
            #     checkpoint_path=checkpoint_path, 
            #     vision_tower=vision_tower,
            #     args=vision_tower_cfg,
            #     patch_size=patch_size,
            #     compression_rate=compression_rate,
            #     lower_bound=lower_bound,
            #     lambda_val=lambda_val,
            #     depth=depth,
            #     num_classes=num_classes,
            #     finetuning_mode=FINETUNING_MODE,
            #     **kwargs)
        
        elif MODE == "Fixed":
            print("🍟" * 20)
            print(f"Using fixed pooling Baseline from the path {checkpoint_path}")
            print("🍟" * 20)

            # vit_loaded: BaselineVisionTower = BaselineVisionTower(
            #     baseline_type=BASELINE_TYPE,
            #     backbone=BACKBONE_NAME,
            #     checkpoint_path=checkpoint_path, 
            #     vision_tower=vision_tower,
            #     args=vision_tower_cfg,
            #     patch_size=patch_size,
            #     compression_rate=compression_rate,
            #     lower_bound=lower_bound,
            #     lambda_val=lambda_val,
            #     depth=depth,
            #     num_classes=num_classes,
            #     finetuning_mode=FINETUNING_MODE,
            #     **kwargs
            # )

            # return vit_loaded

        else:
            vit_loaded: ViTVisionTower = ViTVisionTower(
                checkpoint_path=checkpoint_path, 
                vision_tower=vision_tower,
                args=vision_tower_cfg,
                patch_size=patch_size,
                compression_rate=compression_rate,
                depth=depth,
                finetuning_mode=FINETUNING_MODE,
                **kwargs
            )
            return vit_loaded
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
