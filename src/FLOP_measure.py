# Adapted from:
# https://github.com/raoyongming/DynamicViT/blob/master/calc_flops.py

import warnings
import time
import torch
from numbers import Number
from typing import Any, Callable, List, Optional, Union
from numpy import prod
import numpy as np
from fvcore.nn import FlopCountAnalysis
from open_clip_local.DTP_ViT import DTPViT
from open_clip_local.model import CLIPVisionCfg
from open_clip_local.transformer import VisionTransformer
from open_clip_local.factory import create_model_and_transforms

def register_elemwise_flop_handlers(fca: FlopCountAnalysis):
    def elemwise(i, o):
        try:
            return int(torch.prod(torch.tensor(o[0].shape)).item())
        except:
            return 0
    def native_mha_flops(inputs, outputs):
        q_shape = inputs[0].type().sizes()
        if not q_shape or None in q_shape or len(q_shape) != 3:
            return 0

        seq_len, batch_size, embed_dim = q_shape
        num_heads = 12  # You can make this dynamic if needed

        # FLOPs for QK^T and AV
        flops_qk = 2 * batch_size * num_heads * seq_len * seq_len
        flops_av = 2 * batch_size * num_heads * seq_len * embed_dim

        return flops_qk + flops_av

    # Fallback for common ops
    ops = [
        "aten::add", "aten::sub", "aten::sub_", "aten::rsub", "aten::mul",
        "aten::div", "aten::sigmoid", "aten::gelu", "aten::softmax",
        "aten::sum", "aten::mean", "aten::log", "aten::log1p", "aten::ne",
        "aten::clone", "aten::cumsum", "aten::repeat", "aten::rand",
        "aten::broadcast_tensors", "aten::fill_"
    ]
    for op in ops:
        fca.set_op_handle(op, elemwise)
    fca.set_op_handle("aten::softmax", lambda i, o: 5 * elemwise(i, o))  # heuristic
    fca.set_op_handle("aten::_native_multi_head_attention", native_mha_flops)


@torch.no_grad()
def calc_flops(model: DTPViT | VisionTransformer, img_size: int = 224, show_details: bool = False) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    fca = FlopCountAnalysis(model, dummy_input)
    register_elemwise_flop_handlers(fca)

    total_flops = fca.total()

    print("==================================================")
    print(f"[INFO] Input shape: {tuple(dummy_input.shape)}")
    if isinstance(model, DTPViT):
        ratios = model.compression_rate
        print(f"[INFO] Compression ratio: {ratios}")
    print(f"[INFO] Total GFLOPs: {total_flops / 1e9:.2f}")
    print("==================================================")

    if show_details:
        print("FLOPs by module:")
        print(fca.by_module())

    return total_flops / 1e9

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



# Example usage with a dummy model
if __name__ == "__main__":
    PATCH_SIZE = 16
    COMPRESSION_RATE = 0.5

    cfg = CLIPVisionCfg(
        image_size=224,
        patch_size=PATCH_SIZE,
        width=768,
        mlp_ratio=4.0,
        patch_dropout=0.1,
    )


    model = DTPViT(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        in_chans=3,
        embed_dim=cfg.width,
        depth=(2, 10, 0),            # originally (2, 8, 2) from the DTP paper
        num_heads=cfg.width // 64,  # 768 // 64 = 12
        mlp_ratio=cfg.mlp_ratio,
        drop_rate=cfg.patch_dropout,
        attn_drop_rate=0.1,
        temp=0.5,
        compression_rate=COMPRESSION_RATE,
        threshold=0.5,
        activation_function="gelu",
        num_classes=cfg.width,
        flop_measure=True,
    )

    # Run FLOP measurement
    calc_flops(model, img_size=224, show_details=False)
    
    vit_model = VisionTransformer(
        image_size=224,
        patch_size=PATCH_SIZE,      # or 8, or any divisor of 224
        width=768,                  # hidden size
        layers=12,                  # depth
        heads=12,                   # attention heads
        mlp_ratio=4.0,              # MLP ratio
        output_dim=512              # projection dim, used for CLIP – can be set arbitrarily
    )

    # Run FLOP measurement for ViT
    calc_flops(vit_model, img_size=224, show_details=False)
