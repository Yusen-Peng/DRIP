from transformers import AutoModel
from src.open_clip_local.Qwen2VL_ViT import Qwen2VLViT, Qwen2VLVisionConfig
from transformers.models.siglip.modeling_siglip import SiglipModel


def main():
    

    cfg = Qwen2VLVisionConfig(
        depth=12,
        embed_dim=768,
        num_heads=12,
        patch_size=16,
        temporal_patch_size=1,
        spatial_merge_size=1,   # safest default for plain ViT patch grid
        output_dim=768,         # or whatever you want your proj to do
    )
    
    vit = Qwen2VLViT(cfg)

    vit.load_siglip2_vision_from_full_sd()


if __name__ == "__main__":
    main()
