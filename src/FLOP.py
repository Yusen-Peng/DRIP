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
from open_clip_local.DTP_ViT import DTPViT, DTPViT_Causal, DTPViT_CosSim

from open_clip_local.DTP_ViT import XL_Baseline
from open_clip_local.transformer import VisionTransformer
from open_clip_local.DTP_ViT import DTPViT_Fixed


from open_clip_local.Qwen2VL_ViT import Qwen2VLViT, Qwen2VLVisionConfig, Qwen2VLDRIP


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
        # x = torch.randn(10, 3, img_size, img_size)
        
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

    import argparse
    parser = argparse.ArgumentParser(description='Calculate GFLOPs for different ViT variants')
    # "DRIP" or "DRIP_Causal" or "fixed_pooling" or "ViT" or "XL_baseline" or "Qwen2VL_ViT" or "Qwen2VL_DRIP"
    parser.add_argument('--mode', type=str, default='DRIP', help='Model variant to calculate GFLOPs for')
    parser.add_argument('--compression_rate', type=float, default=0.25, help='Compression rate for DRIP and fixed pooling variants')
    args = parser.parse_args()

    MODE = args.mode
    print(f"Selected mode: {MODE}")
    COMPRESSION_RATE = args.compression_rate
    img_size = 224
    width = 768
    mlp_ratio = 4.0
    patch_dropout = 0.1
    if MODE == "DRIP":
        print(f"🥶🥶🥶🥶Calculating GFLOPs for DRIP with compression rate {COMPRESSION_RATE}...🥶🥶🥶🥶")
        model = DTPViT(
            image_size=img_size,
            patch_size=patch_size,
            width=width,
            layers=12,
            depth=(4, 8, 0),
            compression_rate=COMPRESSION_RATE,
            heads=width // 64,
            mlp_ratio=mlp_ratio,
            temp=0.5,
            output_dim=512,
            pos_embed_type='sin_cos_2d', # 'learnable' or 'sin_cos_2d'
            pool_type='avg',
            flop_measure=True
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
    
    elif MODE == "fixed_pooling":
        print("Calculating GFLOPs for Fixed Pooling...")
        model = DTPViT_Fixed(            
            image_size=img_size,
            patch_size=patch_size,
            width=width,
            layers=12,
            depth=(4, 8, 0),
            compression_rate=COMPRESSION_RATE,
            heads=width // 64,
            mlp_ratio=mlp_ratio,
            temp=0.5,
            output_dim=512,
            pos_embed_type='sin_cos_2d', # 'learnable' or 'sin_cos_2d'
            pool_type='avg',
            flop_measure=True
        )
    elif MODE == "EViT":
        from open_clip_local.EViT import EViT
        print("Calculating GFLOPs for EViT...")
        keep_rate = [1.0] * 12
        keep_rate[3] = COMPRESSION_RATE # only prune at layer 4 (0-indexed, 4+8)
        model = EViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=1000,
            embed_dim=width,
            depth=12,
            num_heads=width // 64,
            mlp_ratio=mlp_ratio,
            qkv_bias=False,
            drop_rate=patch_dropout,
            attn_drop_rate=0.1,
            drop_path_rate=0.1,
            norm_layer=torch.nn.LayerNorm,
            keep_rate=keep_rate
        )

    elif MODE == "EViT_XL":
        from open_clip_local.EViT import EViT_XL_adapted
        print("Calculating GFLOPs for EViT XL...")
        keep_rate = [1.0] * 12
        keep_rate[3] = COMPRESSION_RATE # only prune at layer 4 (0-indexed, 4+8)
        model = EViT_XL_adapted(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=1000,
            embed_dim=width,
            depth=12,
            num_heads=width // 64,
            mlp_ratio=mlp_ratio,
            qkv_bias=False,
            drop_rate=patch_dropout,
            attn_drop_rate=0.1,
            drop_path_rate=0.1,
            norm_layer=torch.nn.LayerNorm,
            keep_rate=keep_rate
        )
    
    elif MODE == 'ToME':
        from open_clip_local.ToME import ToMEViT
        print("Calculating GFLOPs for ToME...")
        # approximately 4x compression at layer 4 and 5
        tome_r_schedule = [0] * 12
        tome_r_schedule[3] = 98  # layer 4
        tome_r_schedule[4] = 49  # layer 5
        model = ToMEViT(
            image_size=img_size,
            patch_size=patch_size,
            width=width,
            layers=12,
            heads=width // 64,
            mlp_ratio=mlp_ratio,
            output_dim=512,
            tome_r_schedule=tome_r_schedule,
            tome_class_token=True,
            tome_distill_token=False,
            tome_use_wavg=True,
            tome_metric="x"
        )
    elif MODE == 'ToME_XL':
        from open_clip_local.ToME import XL_ToMEViT
        print("Calculating GFLOPs for ToME XL...")
        # approximately 4x compression at layer 4 and 5
        tome_r_schedule = [0] * 12
        tome_r_schedule[3] = 98  # layer 4
        tome_r_schedule[4] = 49  # layer 5
        model = XL_ToMEViT(
            image_size=img_size,
            patch_size=patch_size,
            width=width,
            layers=12,
            heads=width // 64,
            mlp_ratio=mlp_ratio,
            output_dim=512,
            tome_r_schedule=tome_r_schedule,
            tome_class_token=True,
            tome_distill_token=False,
            tome_use_wavg=True,
            tome_metric="x"
        )
    elif MODE == "DTEM":
        print("Calculating GFLOPs for DTEM...")
        import timm
        from open_clip_local.DTEM import patch_deit, LogitsOnly

        base = timm.create_model("deit_base_patch16_224", pretrained=False)
        base = patch_deit(base, k2=3, tau1=0.1, tau2=0.1, feat_dim=128)

        # configure r schedule:
        # approximately 4x compression at layer 4 and 5
        r_schedule = [0] * 12
        r_schedule[3] = 98
        r_schedule[4] = 49
        base.update_r(r_schedule)
        model = LogitsOnly(base)
    
    elif MODE == "DTEM_XL":
        print("Calculating GFLOPs for DTEM XL...")
        import timm
        from open_clip_local.DTEM import patch_deit_XL, LogitsOnly

        base = timm.create_model("vit_base_patch16_224", pretrained=False)
        base = patch_deit_XL(base, k2=3, tau1=0.1, tau2=0.1, feat_dim=128)

        # configure r schedule:
        # approximately 4x compression at layer 4 and 5
        r_schedule = [0] * 12
        r_schedule[3] = 98
        r_schedule[4] = 49
        base.update_r(r_schedule)
        model = LogitsOnly(base)

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
