import os
from .clip_encoder import CLIPVisionTowerS2, DRIPVisionTower, ViTVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):

    # FIXME: all hardcoded. Need to be fixed later.
    USE_DTP = False
    FINETUNING_MODE = True
    BACKBONE = 'ViT'  # 'ViT' or 'XL'

    if USE_DTP:
        print("🍟" * 20)
        print("Using DTP-ViT as the vision tower")
        print("🍟" * 20)
    else:
        print("🍟" * 20)
        print("Using original ViT as the vision tower")
        print("🍟" * 20)
    
    #checkpoint_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/ViT_B_16/checkpoints/epoch_15.pt"
    #checkpoint_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_4x_16_ViT_5_7/checkpoints/epoch_15.pt"
    #checkpoint_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_4x_16_ViT_4_8/checkpoints/epoch_15.pt"
    #checkpoint_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_4x_16_ViT_2_10/checkpoints/epoch_15.pt"
    #checkpoint_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_10x_16_ViT_4_8/checkpoints/epoch_15.pt"
    #checkpoint_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_10x_16_ViT_5_7/checkpoints/epoch_15.pt"
    
    #checkpoint_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_10x_16_XL_5_7/checkpoints/epoch_15.pt"
    #checkpoint_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_10x_16_XL_4_8/checkpoints/epoch_15.pt"
    #checkpoint_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_4x_16_XL_4_8/checkpoints/epoch_15.pt"
    #checkpoint_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/CLIP/DRIP_4x_16_XL_2_10/checkpoints/epoch_15.pt"
    
    # trainable vision tower?
    checkpoint_path = "/fs/scratch/PAS2836/yusenpeng_checkpoint/ViT-base-finetune-ALL/vision_tower.pt"


    patch_size = 16
    compression_rate = 0.1
    depth = (5, 7, 0)

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
                backbone=BACKBONE,
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
            print("🍟" * 20)
            print(f"Using original ViT from the path {checkpoint_path}")
            print("🍟" * 20)

            vit_loaded: ViTVisionTower = ViTVisionTower(
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
                **kwargs
            )

            return vit_loaded
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
