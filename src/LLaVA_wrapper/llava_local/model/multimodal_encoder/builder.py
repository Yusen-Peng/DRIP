import os
from .clip_encoder import CLIPVisionTowerS2, CLIPVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):


    ############################################################
    """
        Instructions:
            "ViT": original model checkpoint
            "Fixed": fixed pooling
            "PruMerge": LLaVA-PruMerge
            "ToME": ToME (to be added)
            "DRIP": our design (to be implemented)
    """

    MERGE_STRATEGY = "ViT"
    COMPRESSION_RATE = 0.1
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
                **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
