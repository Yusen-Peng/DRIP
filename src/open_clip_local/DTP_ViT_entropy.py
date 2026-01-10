import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple
from .transformer import LayerNorm, PatchDropout, AttentionalPooler, _expand_token
from .transformer import PositionalEmbedding
from .transformer import TransformerXL
from .pos_embed import get_2d_sincos_pos_embed
from .BP import BoundaryPredictor, downsample
from .utils import to_2tuple
import numpy as np

@torch.no_grad()
def entropy_boundary_mask_torch(
    img: torch.Tensor,
    grid_h: int,
    grid_w: int,
    top_frac: float = 0.25,
    bins: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert img.dim() == 4 and img.size(1) == 3, "img must be [B,3,H,W]"
    B, _, H, W = img.shape

    patch_h = H // grid_h
    patch_w = W // grid_w
    Hc = patch_h * grid_h
    Wc = patch_w * grid_w
    img = img[:, :, :Hc, :Wc]

    # RGB -> grayscale (luma), still float
    gray = 0.2989 * img[:, 0] + 0.5870 * img[:, 1] + 0.1140 * img[:, 2]   # [B, Hc, Wc]
    gray = gray.clamp(0, 1)

    # quantize to [0, bins-1]
    q = (gray * (bins - 1)).round().to(torch.long)  # [B, Hc, Wc]

    # patchify: [B, grid_h, grid_w, patch_h, patch_w] -> [B, N, P]
    patches = q.view(B, grid_h, patch_h, grid_w, patch_w).permute(0, 1, 3, 2, 4)
    patches = patches.reshape(B, grid_h * grid_w, patch_h * patch_w)      # [B, N, P]
    N, P = grid_h * grid_w, patch_h * patch_w

    # histogram via one_hot + sum: [B, N, P, bins] -> [B, N, bins]
    hist = torch.zeros((B, N, bins), device=img.device, dtype=torch.float32)
    ones = torch.ones((B, N, P), device=img.device, dtype=torch.float32)
    hist.scatter_add_(dim=2, index=patches, src=ones)

    # probability per bin
    p = hist / (hist.sum(dim=-1, keepdim=True) + 1e-12)                   # [B, N, bins]

    # entropy: -sum p log2 p
    ent = -(p * (p + 1e-12).log2()).sum(dim=-1)                           # [B, N]
    ent_map = ent.view(B, grid_h, grid_w)

    # top-k mask per image
    k = max(1, int(torch.ceil(torch.tensor(top_frac * N)).item()))
    topk_idx = torch.topk(ent, k=k, dim=1, largest=True).indices          # [B, k]

    hard = torch.zeros((B, N), device=img.device, dtype=torch.float32)
    hard.scatter_(1, topk_idx, 1.0)
    hard_mask = hard.view(B, grid_h, grid_w)

    return ent_map, hard_mask

@torch.no_grad()
def build_entropy_boundaries(
    x: torch.Tensor,            # [B,3,H,W]
    grid_h: int,
    grid_w: int,
    top_frac: float,
) -> torch.Tensor:
    B = x.size(0)
    L = 1 + grid_h * grid_w

    ent_map, hard_mask = entropy_boundary_mask_torch(
        img=x, grid_h=grid_h, grid_w=grid_w, top_frac=top_frac
    )  # hard_mask: [B, gh, gw]

    hard_boundaries = torch.zeros((B, L), device=x.device, dtype=torch.float32)
    hard_boundaries[:, 0] = 1.0
    hard_boundaries[:, 1:] = hard_mask.reshape(B, -1)
    return hard_boundaries, ent_map


class DTPViT(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            depth: tuple[int],
            compression_rate: float,
            heads: int,
            mlp_ratio: float,
            temp: float,
            flop_measure: bool = False,
            threshold: float = 0.5,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.1,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'transformer-xl', # 'learnable' or 'sin_cos_2d' or 'transformer-xl'
            pool_type: str = 'avg',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim
        self.width = width
        self.layers = layers
        self.depth = depth
        self.prior = compression_rate
        self.threshold = threshold
        self.temp = temp
        self.flop_measure = flop_measure
        self.null_token = nn.Parameter(torch.zeros(1, 1, width))
        # NOTE: may be different
        nn.init.normal_(self.null_token, std=0.02)

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            # bias=False # NOTE: False
        )

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1],\
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        elif pos_embed_type == 'transformer-xl':
            self.positional_embedding = PositionalEmbedding(demb=width)
        else:
            raise ValueError
        self.pos_embed_type = pos_embed_type
    
        # pos-aware attention bias terms
        num_heads = heads
        embed_dim = width
        self.r_w_bias = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))
        self.r_r_bias = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = nn.Identity()
        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.down_ln = norm_layer(width)
        # self.boundary_predictor = BoundaryPredictor(
        #     d_model=width,
        #     d_inner=int(width * mlp_ratio),
        #     activation_function="gelu",
        #     temp=temp,
        #     prior=compression_rate,
        #     bp_type='gumbel',
        #     threshold=threshold
        # )

        self.transformer_pre = TransformerXL(
            width,
            self.depth[0],
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            batch_first=False
        )

        self.transformer_post = TransformerXL(
            width,
            self.depth[1],
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            batch_first=False
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))
        self.head = nn.Linear(embed_dim, 1000)


    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.transformer_pre.grad_checkpointing = enable
        self.transformer_post.grad_checkpointing = enable
    
    @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {'positional_embedding', 'class_embedding'}
        return no_wd

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def _embeds(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # shape = [*, dim, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)

        # patch dropout (if active)
        x = self.patch_dropout(x)

        # apply norm before transformer
        x = self.ln_pre(x)
        return x

    def _pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        return pooled, tokens
    
    def encode(self, x: torch.Tensor, return_loss: bool):

        hard_boundaries, _ = build_entropy_boundaries(
            x, self.grid_size[0], self.grid_size[1], self.prior
        )
        
        # print(f'[debug] hard_boundaries sum: {hard_boundaries.sum(dim=1)}')
        # print(f'[debug] hard_boundaries: {hard_boundaries}')
        # print(f"shape check: hard_boundaries: {hard_boundaries.shape}, x: {x.shape}")

        x = self._embeds(x) # [B, 3, H, W] -> [B, L, D]

        # Compute position embeddings
        T = x.size(1)
        pos_seq = torch.arange(T - 1, -1, -1.0, device=x.device, dtype=x.dtype)
        pos_emb = self.positional_embedding(pos_seq)
        x = self.transformer_pre(
            x, 
            pos_emb=pos_emb, 
            r_w_bias=self.r_w_bias, 
            r_r_bias=self.r_r_bias) # [B, L, D] -> [B, L, D]

        hidden: torch.Tensor = self.down_ln(x) # [B, L, D] -> [B, L, D]
        hidden = hidden.transpose(0, 1) # [B, L, D] -> [L, B, D]
        shortened_hidden = downsample(
            boundaries=hard_boundaries,
            hidden=hidden,
            null_group=self.null_token
        ) # [L, B, D] -> [S, B, D]
        shortened_hidden = shortened_hidden.transpose(0, 1) # [S, B, D] -> [B, S, D]

        S = shortened_hidden.size(1)
        new_pos_seq = torch.arange(S - 1, -1, -1.0, device=x.device, dtype=x.dtype)
        new_pos_emb = self.positional_embedding(new_pos_seq)

        features = self.transformer_post(
            shortened_hidden,
            pos_emb=new_pos_emb, 
            r_w_bias=self.r_w_bias, 
            r_r_bias=self.r_r_bias) # [B, S, D] -> [B, S, D]
        
        if return_loss and not self.flop_measure:
            boundary_loss = torch.tensor(0.0, device=x.device) # entropy-based method: boundary loss disabled
            # boundary_loss = self.boundary_predictor.calc_loss(hard_boundaries)
            avg_boundaries_per_batch = hard_boundaries.sum(dim=1).float().mean().item()
            boundary_ratio = avg_boundaries_per_batch / hard_boundaries.size(1)
            return features, boundary_loss, avg_boundaries_per_batch, boundary_ratio
        else:
            return features # [B, S, D]

    def forward(self, x: torch.Tensor, return_loss: bool = False):
        features_out = self.encode(x, return_loss=return_loss) # [B, 3, H, W] -> [B, S, D]

        if return_loss and not self.flop_measure:
            # encode returns tuple (features, loss, avg_boundaries, boundary_ratio)
            tensor, boundary_loss, avg_boundaries_per_batch, boundary_ratio = features_out
        else:
            tensor = features_out
        
        ###################################### original #######################################
        pooled, tokens = self._pool(tensor) # [B, S, D] -> [B, D], [B, S, D]
        pooled = pooled @ self.proj # [B, D] -> [B, output_dim]

        if self.output_tokens:
            return pooled, tokens

        if return_loss and not self.flop_measure:
            return pooled, boundary_loss, avg_boundaries_per_batch, boundary_ratio # [B, output_dim]
        else:
            return pooled # [B, output_dim]


        ###################################### ablation #######################################
        # pool across sequence dimension with mean pooling
        # tensor: [B, S, D]
        # pad_mask = tensor.abs().sum(dim=-1).eq(0)          # [B, S]  True where padded
        # valid_mask = (~pad_mask).float()                  # [B, S]
        # valid_mask_exp = valid_mask.unsqueeze(-1)         # [B, S, 1]

        # x = tensor * valid_mask_exp                       # [B, S, D]
        # sum_x = x.sum(dim=1)                              # [B, D]  sum over sequence
        # valid_counts = valid_mask.sum(dim=1).clamp(min=1e-6).unsqueeze(-1)  # [B, 1]
        # x = sum_x / valid_counts                          # [B, D]

        # logits = self.head(x)                             # [B, num_classes]

        # if return_loss and not self.flop_measure:
        #     return logits, boundary_loss, avg_boundaries_per_batch, boundary_ratio
        # else:
        #     return logits
