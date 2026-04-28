import os
from .clip_encoder import CLIPVisionTowerS2, CLIPVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):


    ############################################################
    """
        Instructions:
            "ViT": original model checkpoint
            "Fixed": fixed pooling
            "PruMerge": LLaVA-PruMerge
            "DRIP": our design
    """

    MERGE_STRATEGY = "DRIP"
    # 2x - 0.5, 4x - 0.25, 8x - 0.125, 10x - 0.1
    COMPRESSION_RATE = 0.25
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_pretrain/drip.bin"
    DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_finetune_train/checkpoint-2661/drip.bin"

    ############################################################


    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if  vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(
                vision_tower=vision_tower, 
                args=vision_tower_cfg, 
                merge_strategy=MERGE_STRATEGY,
                compression_rate=COMPRESSION_RATE,
                drip_weight_path=DRIP_WEIGHT_PATH,
                **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
