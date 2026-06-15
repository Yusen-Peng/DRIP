import os
from .clip_encoder import CLIPVisionTowerS2, CLIPVisionTower, TimmVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):

    """
        Instructions:
            "ViT": original model checkpoint
            "Fixed": fixed pooling
            "PruMerge": LLaVA-PruMerge
            "DRIP": our BP with MLP
            "DRIP-H": our BP with H-Net
    """

    MERGE_STRATEGY = "DRIP"
    # main result: 2x - 0.5, 4x - 0.25, 8x - 0.125, 10x - 0.1
    # limit test: 20x - 0.05, 100x - 0.01, 500x - 0.002
    COMPRESSION_RATE = 0.25

    # FIXME: temperature tuning
    TEMPERATURE = 0.1
    # TEMPERATURE = 0.01
    
    # DRIP_WEIGHT_PATH = None

    """
        4x paths.
    """
    # main experiments    
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_pretrain/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_finetune_train_lora/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_finetune_train_full/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_second_to_last_train_lora/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_second_to_last_train_full/drip.bin"
    
    # ablations
    # -- temperature
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_pretrain_temp001/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_finetune_temp001_train_full/drip.bin"
    # -- H-Net
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_Hnet_ablation_pretrain/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_Hnet_ablation_train_full/drip.bin"
    # -- loss ratio
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_pretrain/drip.bin"    
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_ratio_ablation_train_full/checkpoint-1215/drip.bin"


    # SigLIP 512 experiments
    DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_pretrain_512_DRIP_4x/drip.bin"


    # Qwen experiments
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B_DRIP_4x_pretrain/drip.bin"



    """
        8x paths.
    """
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_8x_pretrain/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_8x_finetune_train_lora/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_8x_finetune_train_full/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_8x_second_to_last_train_lora/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_8x_second_to_last_train_full/drip.bin"



    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B_DRIP_8x_pretrain/drip.bin"

    """
        10x paths.
    """
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_10x_pretrain/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_10x_finetune_train_lora/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_10x_finetune_train_full/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_10x_second_to_last_train_lora/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_10x_second_to_last_train_full/drip.bin"

    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B_DRIP_10x_pretrain/drip.bin"



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
                temperature=TEMPERATURE,
                **kwargs)
    
    elif vision_tower.startswith("timm/"):
        if use_s2:
            raise NotImplementedError("S2 is not implemented for TimmVisionTower yet.")
        else:
            return TimmVisionTower(
                vision_tower=vision_tower,
                args=vision_tower_cfg,
                merge_strategy=MERGE_STRATEGY,
                compression_rate=COMPRESSION_RATE,
                drip_weight_path=DRIP_WEIGHT_PATH,
                temperature=TEMPERATURE,
                **kwargs
            )
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
