# faithfully adapted from:
# 1) https://github.com/raoyongming/DynamicViT/blob/master/calc_flops.py and
# 2) https://github.com/raoyongming/DynamicViT/blob/master/models/dylvvit.py

import warnings
import time
import torch
from numbers import Number
from typing import Any, List
import numpy as np
from fvcore.nn import FlopCountAnalysis
from open_clip_local.DTP_ViT import DTPViT, HierarchicalDTPViT, SoftDTPViT, XL_Baseline
from open_clip_local.transformer import VisionTransformer

DROPOUT_FLOPS = 4
LAYER_NORM_FLOPS = 5
ACTIVATION_FLOPS = 8
SOFTMAX_FLOPS = 5

def rfft_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the rfft/rfftn operator.
    """
    input_shape = inputs[0].type().sizes()
    B, H, W, C = input_shape
    N = H * W
    flops = N * C * np.ceil(np.log2(N))
    return flops

def calc_flops(model, img_size=224, show_details=False, ratios=None):
    with torch.no_grad():
        x = torch.randn(1, 3, img_size, img_size)
        # model.default_ratio = ratios # this seems useless
        fca1 = FlopCountAnalysis(model, x)
        handlers = {
            'aten::fft_rfft2': rfft_flop_jit,
            'aten::fft_irfft2': rfft_flop_jit,
        }
        fca1.set_op_handle(**handlers)
        flops1 = fca1.total()
        if show_details:
            print(fca1.by_module())
    return flops1 / 1e9

@torch.no_grad()
def throughput(images, model):
    model.eval()
    images = images
    batch_size = images.shape[0]
    for _ in range(50):
        model(images) # warm-up
    print(f"throughput averaged with 30 times")
    tic1 = time.time()
    for _ in range(30):
        model(images)
    tic2 = time.time()
    print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)} images/sec")
    # MB = 1024.0 * 1024.0
    # print('memory:', torch.cuda.max_memory_allocated() / MB)


def main():
    patch_size = 16
    MODE = "DRIP" # "DRIP", "H-DRIP", "S-DRIP","ViT", 'XL_Baseline', "Swin", "DynamicViT", "EViT"

    img_size = 224
    width = 768
    mlp_ratio = 4.0
    patch_dropout = 0.1
    if MODE == "DRIP":
        COMPRESSION_RATE = 0.1  # e.g., 0.1 means keeping 10% patches
        print(f"🥶🥶🥶🥶Calculating GFLOPs for DRIP with compression rate {COMPRESSION_RATE}...🥶🥶🥶🥶")
        model = DTPViT(
            image_size=img_size,
            patch_size=patch_size,
            width=width,
            layers=12,
            depth=(2, 10, 0),
            compression_rate=COMPRESSION_RATE,
            heads=width // 64,
            mlp_ratio=mlp_ratio,
            temp=0.5,
            flop_measure=True
        )
    elif MODE == "H-DRIP":
        rate1 = 0.25  # compression rate at stage 1
        rate2 = 0.25  # compression rate at stage 2
        rate3 = 0.25  # compression rate at stage 3
        model = HierarchicalDTPViT(
            image_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            depth=(2, 2, 6, 2),
            num_heads=[3, 6, 12, 24],
            mlp_ratio=mlp_ratio,
            drop_rate=patch_dropout,
            attn_drop_rate=0.0,
            temp=0.5,
            compression_rate=(rate1, rate2, rate3),  # compression at stage 1 and 2
            threshold=0.5,
            activation_function="gelu",
            num_classes=width,
            flop_measure=True,
        )
    elif MODE == "S-DRIP":
        upper_bound = 0.3  # compression rate upper bound
        lower_bound = 0.2  # compression rate lower bound
        compression_rate = (lower_bound, upper_bound)
        model = SoftDTPViT(
            image_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=width,
            depth=(2, 10, 0),
            num_heads=width // 64,
            mlp_ratio=mlp_ratio,
            drop_rate=patch_dropout,
            attn_drop_rate=0.1,
            temp=0.5,
            compression_rate=compression_rate,
            threshold=0.5,
            activation_function="gelu",
            num_classes=width,
            flop_measure=True,  # simulating fake boundaries for reproducible GFLOPs
        )
    elif MODE == "ViT":
        model = VisionTransformer(
            image_size=img_size,
            patch_size=patch_size,
            width=width,
            layers=12,
            heads=width // 64,
            mlp_ratio=mlp_ratio,
            output_dim=512
        )
    elif MODE == "XL_Baseline":
        model = XL_Baseline(
            image_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=width,
            num_heads=width // 64,
            mlp_ratio=mlp_ratio,
            drop_rate=patch_dropout,
            attn_drop_rate=0.1,
            temp=0.5,
            threshold=0.5,
            activation_function="gelu",
            num_classes=width,
            flop_measure=True,  # simulating fake boundaries for reproducible GFLOPs
        )

    elif MODE == "Swin":
        from swin import SwinTransformer
        print("Calculating GFLOPs for Swin Transformer...")
        model = SwinTransformer(
            img_size=224, 
            patch_size=4, 
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7, mlp_ratio=4.0, qkv_bias=True,
            drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1,
            norm_layer=torch.nn.LayerNorm, ape=False, patch_norm=True,
            use_checkpoint=False, fused_window_process=False,
        )
    
    elif MODE == "adapted_Swin":
        from swin import HierarchicalAdaptedSwin
        print("Calculating GFLOPs for Adapted Swin Transformer...")
        model = HierarchicalAdaptedSwin(
            image_size=224, 
            patch_size=8, 
            in_chans=3,
            embed_dim=96,
            depth=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            drop_rate=0.0, 
            num_classes=768,
            norm_layer=torch.nn.LayerNorm
        )
    
    elif MODE == "single_Swin":
        from swin import SingleAdaptedSwin
        print("Calculating GFLOPs for Single-stage Adapted Swin Transformer...")
        model = SingleAdaptedSwin(
            image_size=224, 
            patch_size=16, 
            in_chans=3,
            embed_dim=768,
            depth=(4, 8),
            num_heads=(12, 12),
            mlp_ratio=4.0,
            drop_rate=0.0, 
            num_classes=768,
            norm_layer=torch.nn.LayerNorm
        )

    elif MODE == "DynamicViT":
        from dynamicViT import VisionTransformerDiffPruning
        print("Calculating GFLOPs for DynamicViT...")
        model = VisionTransformerDiffPruning(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=1000,
            embed_dim=width,
            depth=12, # total 12 layers - keep everything consistent
            num_heads=width // 64,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_rate=patch_dropout,
            attn_drop_rate=0.1,
            drop_path_rate=0.1,
            norm_layer=torch.nn.LayerNorm,
            pruning_loc=[2],
            token_ratio=[0.25], # keep 25% patches at the only pruning location
            distill=False,
            training=False, # for GFLOPs calculation we set it to False to avoid randomness
        )
    elif MODE == "EViT":
        from EViT import EViT
        print("Calculating GFLOPs for EViT...")
        r = 0.25  # keep 25% of patch tokens at block 2 only
        keep_rate = [1.0] * 12
        keep_rate[2] = r
        model = EViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=1000,
            embed_dim=width,
            depth=12,  # total 12 layers - keep everything consistent
            num_heads=width // 64,
            mlp_ratio=mlp_ratio,
            qkv_bias=False,
            drop_rate=patch_dropout,
            attn_drop_rate=0.1,
            drop_path_rate=0.1,
            norm_layer=torch.nn.LayerNorm,
            keep_rate=keep_rate
        )
    else:
        raise NotImplementedError("MODE not implemented")
            

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    flops = calc_flops(model, img_size)
    print('GFLOPs for {}: {}'.format(MODE, round(flops, 2)))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'number of parameters: {round(n_parameters, 2)} M')

    # # throughput test
    # batch_size = 512 # for consistency
    # x = torch.randn(batch_size, 3, img_size, img_size).to(device)
    # throughput(x, model)



if __name__ == "__main__":
    main()
