import torch.nn as nn

def naive_weight_transfer(dtp_vit: nn.Module, clip_vit_state_dict):
    dtp_state_dict = dtp_vit.state_dict()
    transferred = 0

    # map positional embedding
    if "pos_embed" in dtp_state_dict and "positional_embedding" in clip_vit_state_dict:
        dtp_state_dict["pos_embed"].copy_(clip_vit_state_dict["positional_embedding"])
        print("✅ Transferred: pos_embed")

    # map patch embedding
    if "patch_embed.proj.weight" in dtp_state_dict and "conv1.weight" in clip_vit_state_dict:
        dtp_state_dict["patch_embed.proj.weight"].copy_(clip_vit_state_dict["conv1.weight"])
        dtp_state_dict["patch_embed.proj.bias"].zero_()
        print("✅ Transferred: patch_embed.proj")

    # map transformer blocks
    for i in range(12):
        if i < len(dtp_vit.pre_blocks):
            prefix = f"pre_blocks.{i}"
        else:
            j = i - len(dtp_vit.pre_blocks)
            prefix = f"shorten_blocks.{j}"

        base = f"transformer.resblocks.{i}"

        pairs = [
            ("ln_1.weight", "norm1.weight"),
            ("ln_1.bias", "norm1.bias"),
            ("attn.in_proj_weight", "attn.in_proj_weight"),
            ("attn.in_proj_bias", "attn.in_proj_bias"),
            ("attn.out_proj.weight", "attn.out_proj.weight"),
            ("attn.out_proj.bias", "attn.out_proj.bias"),
            ("ln_2.weight", "norm2.weight"),
            ("ln_2.bias", "norm2.bias"),
            ("mlp.c_fc.weight", "mlp.0.weight"),
            ("mlp.c_fc.bias", "mlp.0.bias"),
            ("mlp.c_proj.weight", "mlp.3.weight"),
            ("mlp.c_proj.bias", "mlp.3.bias"),
        ]

        for clip_key, dtp_key in pairs:
            full_clip_key = f"{base}.{clip_key}"
            full_dtp_key = f"{prefix}.{dtp_key}"
            if full_clip_key in clip_vit_state_dict and full_dtp_key in dtp_state_dict:
                if dtp_state_dict[full_dtp_key].shape == clip_vit_state_dict[full_clip_key].shape:
                    dtp_state_dict[full_dtp_key].copy_(clip_vit_state_dict[full_clip_key])
                    transferred += 1
                else:
                    print(f"⚠️ Shape mismatch: {full_clip_key} vs {full_dtp_key}")
                    exit(1)
            else:
                print(f"⛔ Missing: {full_clip_key} or {full_dtp_key}")
                exit(1)
        print("✅ Transferred: " + prefix)

    print(f"🎉 Transferred {transferred} parameter tensors successfully!")
    print(f"DTP-ViT now has {sum(p.numel() for p in dtp_vit.parameters())} parameters.")