import torch
import contextlib
import io
import time
from fvcore.nn import FlopCountAnalysis
from open_clip_local.DTP_ViT import DTPViT
from open_clip_local.model import CLIPVisionCfg
from open_clip_local.transformer import VisionTransformer

def register_custom_flop_handlers(fca):
    def elemwise(i, o):
        shape = i[0].type().sizes()
        if shape is None:
            return 0
        return int(torch.prod(torch.tensor(shape)))

    fca.set_op_handle("aten::add", elemwise)
    fca.set_op_handle("aten::sub", elemwise)
    fca.set_op_handle("aten::sub_", elemwise)
    fca.set_op_handle("aten::rsub", elemwise)
    fca.set_op_handle("aten::mul", elemwise)
    fca.set_op_handle("aten::div", elemwise)
    fca.set_op_handle("aten::sigmoid", elemwise)
    fca.set_op_handle("aten::gelu", elemwise)
    fca.set_op_handle("aten::softmax", lambda i, o: 5 * elemwise(i, o))
    fca.set_op_handle("aten::sum", elemwise)
    fca.set_op_handle("aten::mean", elemwise)
    fca.set_op_handle("aten::log", elemwise)
    fca.set_op_handle("aten::log1p", elemwise)
    fca.set_op_handle("aten::ne", elemwise)
    fca.set_op_handle("aten::clone", elemwise)
    fca.set_op_handle("aten::cumsum", elemwise)
    fca.set_op_handle("aten::repeat", elemwise)
    fca.set_op_handle("aten::rand", elemwise)
    fca.set_op_handle("aten::new_ones", lambda i, o: 0)
    

class FLOPWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, flop_mode=True)


@torch.no_grad()
def calc_flops(model, img_size=224, compression_rate=None, show_details=True):
    seed = 42
    torch.manual_seed(seed)
    dummy_input = torch.randn(1, 3, img_size, img_size)

    fca = FlopCountAnalysis(model, dummy_input)
    register_custom_flop_handlers(fca)
    with contextlib.redirect_stdout(io.StringIO()):
        flops = fca.total()

    if show_details:
        print("FLOPs by module:")
        print(fca.by_module())

    if compression_rate is not None:
        print(f"\n[SUMMARY] Compression Rate: {compression_rate}")
        print(f"GFLOPs: {flops / 1e9:.2f}")
    else:
        print(f"\n[SUMMARY] GFLOPs: {flops / 1e9:.2f}")
    return flops / 1e9


def build_dtpvit():
    cfg = CLIPVisionCfg(
        image_size=224,
        patch_size=32,
        width=768,
        mlp_ratio=4.0,
        patch_dropout=0.1,
    )

    model = DTPViT(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        in_chans=3,
        embed_dim=cfg.width,
        depth=(2, 8, 2),
        num_heads=cfg.width // 64,
        mlp_ratio=cfg.mlp_ratio,
        drop_rate=cfg.patch_dropout,
        attn_drop_rate=0.1,
        temp=0.5,
        compression_rate=0.1,
        threshold=0.5,
        activation_function="gelu",
        num_classes=cfg.width,
    )

    wrapped_model = FLOPWrapper(model.eval())
    return wrapped_model


if __name__ == "__main__":
    model = build_dtpvit()
    img_size = 224
    compression_rate = 0.1
    calc_flops(
        model=model,
        img_size=img_size,
        compression_rate=compression_rate,
        show_details=False
    )

    # now compare with the original ViT
    # ViT-B-32
    vision_cfg = CLIPVisionCfg(
        image_size=224,
        patch_size=32,
        width=768,
        layers=12,
        mlp_ratio=4.0,
        ls_init_value=None,
        patch_dropout=0.1,
        attentional_pool=False,
        attn_pooler_queries=0,
        attn_pooler_heads=0,
        pos_embed_type='learnable',
        no_ln_pre=False,
        final_ln_after_pool=False,
        pool_type='avg',
        output_tokens=False,
    )

    embed_dim = vision_cfg.width
    act_layer = torch.nn.GELU
    norm_layer = torch.nn.LayerNorm

    model = VisionTransformer(
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        width=vision_cfg.width,
        layers=vision_cfg.layers,
        heads=12,
        mlp_ratio=vision_cfg.mlp_ratio,
        ls_init_value=vision_cfg.ls_init_value,
        patch_dropout=vision_cfg.patch_dropout,
        attentional_pool=vision_cfg.attentional_pool,
        attn_pooler_queries=vision_cfg.attn_pooler_queries,
        attn_pooler_heads=vision_cfg.attn_pooler_heads,
        pos_embed_type=vision_cfg.pos_embed_type,
        no_ln_pre=vision_cfg.no_ln_pre,
        final_ln_after_pool=vision_cfg.final_ln_after_pool,
        pool_type=vision_cfg.pool_type,
        output_tokens=vision_cfg.output_tokens,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    wrapped_model = model.eval()

    calc_flops(
        model=wrapped_model,
        img_size=img_size,
        compression_rate=None,
        show_details=False
    )
