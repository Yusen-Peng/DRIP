import os
from .clip_encoder import CLIPVisionTowerS2, CLIPVisionTower, TimmVisionTower
from .siglip_encoder import SiglipVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):

    """
        Instructions:
            "ViT": original model checkpoint
            "Fixed": fixed pooling
            "PruMerge": LLaVA-PruMerge (ICCV 2025)
            "DRIP": our BP with MLP
            "DRIP-H": our BP with H-Net
            "PruneSID": PruneSID (ICLR 2026)
            "Perceiver": Perceiver (ICML 2021)
    """
    # NOTE: This is irrelevant for CLIP-based models,
    # but important for evaluating timm models.
    INFERENCE_MODE = True


    MERGE_STRATEGY = "DRIP"
    # main result: 2x - 0.5, 4x - 0.25, 8x - 0.125, 10x - 0.1
    # limit test: 20x - 0.05, 100x - 0.01, 500x - 0.002
    COMPRESSION_RATE = 0.25

    # NOTE: temperature tuning
    TEMPERATURE = 0.1

    # TEMPERATURE = 0.01
    # TEMPERATURE = 0.3
    # TEMPERATURE = 0.5
    # TEMPERATURE = 0.8
    # TEMPERATURE = 1.0
    # TEMPERATURE = 1.5
    # TEMPERATURE = 2.0

    DRIP_WEIGHT_PATH = None
    PERCEIVER_WEIGHT_PATH = None

    DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_4x_pretrain_temp01_new_downsample/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_4x_pretrain_temp15_new_downsample/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_4x_pretrain_temp20_new_downsample/drip.bin"



    """
        4x paths.
    """
    # main experiments    
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_pretrain/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_finetune_train_lora/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_finetune_train_full/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_second_to_last_train_lora/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_second_to_last_train_full/drip.bin"    

    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Perceiver_4x_pretrain/perceiver.bin"
    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Perceiver_4x_train_all/perceiver.bin"
    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Perceiver_4x_pretrain_second/perceiver.bin"
    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Perceiver_4x_train_all_second/perceiver.bin"


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


    # Qwen experiments
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_DRIP_4x_pretrain/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_DRIP_4x_train_full/drip.bin"


    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_Perceiver_4x_pretrain/perceiver.bin"
    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_Perceiver_4x_train_full/perceiver.bin"



    # SigLIP experiments
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_4x_pretrain/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_4x_train_full/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_4x_pretrain_temp08_new_downsample/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_4x_temp08_new_downsample_train_full/drip.bin"

    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_4x_pretrain_temp10_new_downsample/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_4x_temp10_new_downsample_train_full/drip.bin"



    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_Perceiver_4x_pretrain/perceiver.bin"
    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_Perceiver_4x_train_full/perceiver.bin"


    """
        8x paths.
    """
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_8x_pretrain/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_8x_finetune_train_lora/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_8x_finetune_train_full/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_8x_second_to_last_train_lora/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_8x_second_to_last_train_full/drip.bin"


    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Perceiver_8x_pretrain/perceiver.bin"
    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Perceiver_8x_train_all/perceiver.bin"
    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Perceiver_8x_pretrain_second/perceiver.bin"
    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Perceiver_8x_train_all_second/perceiver.bin"


    # qwen experiments
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_DRIP_8x_pretrain/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_DRIP_8x_train_full/drip.bin"


    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_Perceiver_8x_pretrain/perceiver.bin"
    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_Perceiver_8x_train_full/perceiver.bin"


    # SigLIP experiments
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_8x_pretrain/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_8x_train_full/drip.bin"
    
    
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_8x_pretrain_temp08_new_downsample/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_8x_temp08_new_downsample_train_full/drip.bin"

    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_8x_pretrain_temp05_new_downsample/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_8x_temp05_new_downsample_train_full/drip.bin"

    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_8x_pretrain_temp10_new_downsample/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_8x_temp10_new_downsample_train_full/drip.bin"



    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_Perceiver_8x_pretrain/perceiver.bin"
    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_Perceiver_8x_train_full/perceiver.bin"


    """
        10x paths.
    """
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_10x_pretrain/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_10x_finetune_train_lora/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_10x_finetune_train_full/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_10x_second_to_last_train_lora/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_10x_second_to_last_train_full/drip.bin"


    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Perceiver_10x_pretrain/perceiver.bin"
    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Perceiver_10x_train_all/perceiver.bin"
    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Perceiver_10x_pretrain_second/perceiver.bin"
    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_Perceiver_10x_train_all_second/perceiver.bin"

    
    
    # Qwen experiments
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_DRIP_10x_pretrain/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_DRIP_10x_train_full/drip.bin"

    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_Perceiver_10x_pretrain/perceiver.bin"
    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_Qwen2.5-14B-Instruct_Perceiver_10x_train_full/perceiver.bin"



    # SigLIP experiments
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_10x_pretrain/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_10x_train_full/drip.bin"

    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_10x_pretrain_temp001/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_10x_train_full_temp001/drip.bin"


    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_10x_pretrain_temp05/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_10x_pretrain_temp08/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_10x_pretrain_temp10/drip.bin"


    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_10x_train_full_temp05/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_10x_train_full_temp08/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_10x_train_full_temp10/drip.bin"


    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_10x_pretrain_temp01_new_downsample/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_10x_pretrain_temp08_new_downsample/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_10x_temp01_new_downsample_train_full/drip.bin"

    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_10x_temp08_new_downsample_train_full/drip.bin"

    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_10x_pretrain_temp08_new_downsample_LEADING_ONE/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_10x_temp08_new_downsample_LEADING_ONE_train_full/drip.bin"


    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_10x_pretrain_temp10_new_downsample/drip.bin"
    # DRIP_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_DRIP_10x_temp10_new_downsample_train_full/drip.bin"




    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_Perceiver_10x_pretrain/perceiver.bin"
    # PERCEIVER_WEIGHT_PATH = "/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_SigLIP_HF_v2_Perceiver_10x_train_full/perceiver.bin"



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
                perceiver_weight_path=PERCEIVER_WEIGHT_PATH,
                temperature=TEMPERATURE,
                **kwargs)
    elif vision_tower.startswith("google/"):
        if use_s2:
            raise NotImplementedError("S2 is not implemented for GoogleVisionTower yet.")
        else:
            return SiglipVisionTower(
                vision_tower=vision_tower,
                args=vision_tower_cfg,
                merge_strategy=MERGE_STRATEGY,
                compression_rate=COMPRESSION_RATE,
                drip_weight_path=DRIP_WEIGHT_PATH,
                perceiver_weight_path=PERCEIVER_WEIGHT_PATH,
                temperature=TEMPERATURE,
                **kwargs
            )
        
    
    elif vision_tower.startswith("timm/"):
        if use_s2:
            raise NotImplementedError("S2 is not implemented for TimmVisionTower yet.")
        else:
            if INFERENCE_MODE:
                kwargs.pop("delay_load", None)  # remove duplicate
                return TimmVisionTower(
                    vision_tower=vision_tower,
                    args=vision_tower_cfg,
                    merge_strategy=MERGE_STRATEGY,
                    compression_rate=COMPRESSION_RATE,
                    drip_weight_path=DRIP_WEIGHT_PATH,
                    temperature=TEMPERATURE,
                    delay_load=False, # don't delay for Timm models for proper inference
                    **kwargs
                )
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
