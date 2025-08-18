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
from open_clip_local.DTP_ViT import DTPViT, HierarchicalDTPViT, SoftDTPViT
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

    images = images.cuda(non_blocking=True)
    batch_size = images.shape[0]
    for i in range(50):
        model(images)
    torch.cuda.synchronize()
    print(f"throughput averaged with 30 times")
    tic1 = time.time()
    for i in range(30):
        model(images)
    torch.cuda.synchronize()
    tic2 = time.time()
    print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
    MB = 1024.0 * 1024.0
    print('memory:', torch.cuda.max_memory_allocated() / MB)


def main():
    patch_size = 16
    MODE = "DRIP" # "DRIP", "H-DRIP", "S-DRIP","ViT"

    img_size = 224
    width = 768
    mlp_ratio = 4.0
    patch_dropout = 0.1
    if MODE == "DRIP":
        COMPRESSION_RATE = 0.25
        model = DTPViT(
            image_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=width,
            depth=(4, 8, 0),
            num_heads=width // 64,
            mlp_ratio=mlp_ratio,
            drop_rate=patch_dropout,
            attn_drop_rate=0.1,
            temp=0.5,
            compression_rate=COMPRESSION_RATE,
            threshold=0.5,
            activation_function="gelu",
            num_classes=width,
            flop_measure=True, # simulating fake boundaries for reproducible GFLOPs
        )
    elif MODE == "H-DRIP":
        rate1 = 0.5  # compression rate at stage 1
        rate2 = 0.5  # compression rate at stage 2
        model = HierarchicalDTPViT(
            image_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=width,
            depth=(3, 3, 6),
            num_heads=width // 64,
            mlp_ratio=mlp_ratio,
            drop_rate=patch_dropout,
            attn_drop_rate=0.1,
            temp=0.5,
            compression_rate=(rate1, rate2),  # compression at stage 1 and 2
            threshold=0.5,
            activation_function="gelu",
            num_classes=width,
            flop_measure=True,  # simulating fake boundaries for reproducible GFLOPs
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
    else:
        raise ValueError("Unknown model mode: {}".format(MODE))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    flops = calc_flops(model, img_size)
    print('GFLOPs for {}: {}'.format(MODE, round(flops, 2)))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'number of parameters: {round(n_parameters, 2)} M')


if __name__ == "__main__":
    main()
