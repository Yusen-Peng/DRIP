"""
    Code adapted from: https://github.com/facebookresearch/ToMe/blob/main/tome/merge.py
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import math
from typing import Callable, Tuple
import torch
import torch.nn as nn
from typing import Optional, OrderedDict
from typing import Callable, Tuple, Optional, List, Union
import torch.nn.functional as F
from .transformer import LayerNorm, Attention, LayerScale, PatchDropout, AttentionalPooler, _expand_token
from .utils import to_2tuple
from .pos_embed import get_2d_sincos_pos_embed



def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def kth_bipartite_soft_matching(
    metric: torch.Tensor, k: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (every kth element, the rest).
    If n is the number of tokens, resulting number of tokens will be n // z.

    Input size is [batch, tokens, channels].
    z indicates the stride for the first set.
    z = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
    """
    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        return a, b

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        r = a.shape[1]
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def random_bipartite_soft_matching(
    metric: torch.Tensor, r: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (r chosen randomly, the rest).
    Input size is [batch, tokens, channels].

    This will reduce the number of tokens by r.
    """
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        B, N, _ = metric.shape
        rand_idx = torch.rand(B, N, 1, device=metric.device).argsort(dim=1)

        a_idx = rand_idx[:, :r, :]
        b_idx = rand_idx[:, r:, :]

        def split(x):
            C = x.shape[-1]
            a = x.gather(dim=1, index=a_idx.expand(B, r, C))
            b = x.gather(dim=1, index=b_idx.expand(B, N - r, C))
            return a, b

        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        C = src.shape[-1]
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        C = x.shape[-1]
        dst = x
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, C))

        out = torch.zeros(B, N, C, device=x.device, dtype=x.dtype)

        out.scatter_(dim=-2, index=a_idx.expand(B, r, C), src=src)
        out.scatter_(dim=-2, index=b_idx.expand(B, N - r, C), src=dst)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source


"""
    Below is our own integration of ToME into ViT blocks.
"""

class ToMECustomResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        scale_cosine_attn: bool = False,
        scale_heads: bool = False,
        scale_attn: bool = False,
        scale_fc: bool = False,
        batch_first: bool = True,
        tome_r: int = 0,
        tome_class_token: bool = True,
        tome_distill_token: bool = False,
        tome_use_wavg: bool = True,
        tome_metric: str = "x", # "x" or "ln2" (metric to compute similarity)
    ):
        super().__init__()

        # original block parts from ``CustomResidualAttentionBlock``
        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model,
            n_head,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
            batch_first=batch_first,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()


        # ToMe params
        self.tome_r = int(tome_r)
        self.tome_class_token = tome_class_token
        self.tome_distill_token = tome_distill_token
        self.tome_use_wavg = tome_use_wavg
        assert tome_metric in ("x", "ln2")
        self.tome_metric = tome_metric
        # token size tracker for weighted merges
        self.register_buffer("_tome_size", None, persistent=False)

    def get_reference_weight(self):
        return self.mlp.c_fc.weight

    def _get_size(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        if self._tome_size is None or self._tome_size.shape[:2] != x.shape[:2]:
            self._tome_size = torch.ones(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)
        return self._tome_size

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # Attention residual
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))

        # ToMe merging step
        if self.tome_r > 0:
            metric = x if self.tome_metric == "x" else self.ln_2(x)
            merge, _ = bipartite_soft_matching(
                metric=metric,
                r=self.tome_r,
                class_token=self.tome_class_token,
                distill_token=self.tome_distill_token,
            )

            if self.tome_use_wavg:
                size = self._get_size(x)
                x, size = merge_wavg(merge, x, size=size)
                self._tome_size = size
            else:
                x = merge(x, mode="mean")
                self._tome_size = None

        # MLP residual
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class ToMETransformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        batch_first: bool = True,
        tome_r_schedule=0, # ToMe schedule: list length=layers or scalar
        tome_class_token: bool = True,
        tome_distill_token: bool = False,
        tome_use_wavg: bool = True,
        tome_metric: str = "x",
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first
        self.grad_checkpointing = False

        if isinstance(tome_r_schedule, int):
            tome_r_schedule = [tome_r_schedule] * layers
        assert len(tome_r_schedule) == layers

        self.resblocks = nn.ModuleList([
            ToMECustomResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                batch_first=batch_first,
                tome_r=int(tome_r_schedule[i]),
                tome_class_token=tome_class_token,
                tome_distill_token=tome_distill_token,
                tome_use_wavg=tome_use_wavg,
                tome_metric=tome_metric,
            )
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()

        for blk in self.resblocks:
            x = blk(x, attn_mask=attn_mask)

        if not self.batch_first:
            x = x.transpose(0, 1)
        return x


class ToMEViT(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        ls_init_value: float = None,
        attentional_pool: bool = False,
        attn_pooler_queries: int = 256,
        attn_pooler_heads: int = 8,
        output_dim: int = 512,
        patch_dropout: float = 0.,
        no_ln_pre: bool = False,
        pos_embed_type: str = 'learnable',
        pool_type: str = 'tok',
        final_ln_after_pool: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        output_tokens: bool = False,

        # ToMe explicit params
        tome_r_schedule: Union[int, List[int]] = 0, # int or list length=layers
        tome_class_token: bool = True,
        tome_distill_token: bool = False,
        tome_use_wavg: bool = True,
        tome_metric: str = "x", # "x" or "ln2"
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        assert tome_metric in ("x", "ln2")

        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim
        self.width = width

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width)
            )
        elif pos_embed_type == 'sin_cos_2d':
            assert self.grid_size[0] == self.grid_size[1], \
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width),
                requires_grad=False
            )
            pe = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pe).float())
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()
        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)

        self.transformer = ToMETransformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            batch_first=True,

            tome_r_schedule=tome_r_schedule,
            tome_class_token=tome_class_token,
            tome_distill_token=tome_distill_token,
            tome_use_wavg=tome_use_wavg,
            tome_metric=tome_metric,
        )

        # pooling head (unchanged)
        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim, width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim, width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    raise ValueError(f"Unknown attentional_pool type: {attentional_pool}")
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim, width,
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

        self.init_parameters()

    def init_parameters(self):
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.transformer.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'positional_embedding', 'class_embedding'}

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x
        return pooled, tokens

    def _embeds(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)                 # [B, C, Gh, Gw]
        x = x.reshape(x.shape[0], x.shape[1], -1)   # [B, C, Gh*Gw]
        x = x.permute(0, 2, 1)           # [B, Gh*Gw, C]

        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        return x

    def _pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.attn_pool is not None:
            if getattr(self, "attn_pool_contrastive", None) is not None:
                x = self.ln_post(x)
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    pooled = self.attn_pool_contrastive(tokens)
            else:
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

    def encode(self, x: torch.Tensor):
        x = self._embeds(x)
        x = self.transformer(x)
        return x

    def forward(self, x: torch.Tensor):
        x = self._embeds(x)
        x = self.transformer(x)
        pooled, tokens = self._pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens
        return pooled


class XLAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        batch_first: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        max_len: int = 1024,   # must be >= max token length you will ever have
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.batch_first = batch_first

        # match your weight scheme
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3)) if qkv_bias else None

        # Transformer-XL bits
        self.r_net = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.r_w_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.r_r_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

        self.max_len = max_len
        self.rel_emb = nn.Embedding(2 * max_len - 1, dim)
        nn.init.normal_(self.rel_emb.weight, mean=0.0, std=0.02)

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, Q, K]
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3)                 # [B,H,Q,K+1]
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))  # [B,H,K+1,Q]
        x = x_padded[:, :, 1:, :].view_as(x)                       # [B,H,Q,K]
        return x

    def _build_r(self, klen: int, device: torch.device) -> torch.Tensor:
        if klen > self.max_len:
            raise ValueError(f"klen={klen} > max_len={self.max_len} (increase max_len)")
        # use distances: (klen-1 .. 0) mapped into embedding indices
        dist = torch.arange(klen - 1, -1, -1, device=device)       # [klen]
        idx = (self.max_len - 1) + dist                            # shift into [0 .. 2*max_len-2]
        r = self.rel_emb(idx)                                      # [klen, dim]
        return r                                                   # [klen, dim]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # x: [B,T,C] if batch_first else [T,B,C]
        if self.batch_first:
            x = x.transpose(0, 1)  # [T,B,C]

        T, B, C = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)   # [T,B,3C]
        q, k, v = qkv.chunk(3, dim=-1)

        # [T,B,H,D]
        q = q.view(T, B, self.num_heads, self.head_dim)
        k = k.view(T, B, self.num_heads, self.head_dim)
        v = v.view(T, B, self.num_heads, self.head_dim)

        # build r and project
        r = self._build_r(T, x.device)                              # [T, C]
        r_head_k = self.r_net(r).view(T, self.num_heads, self.head_dim)  # [T,H,D]

        # AC term
        rw_head_q = q + self.r_w_bias                                # broadcast [H,D]
        AC = torch.einsum("tbhd,sbhd->bhts", rw_head_q, k)           # [B,H,T,T]

        # BD term
        rr_head_q = q + self.r_r_bias
        BD = torch.einsum("tbhd,shd->bhts", rr_head_q, r_head_k)     # [B,H,T,T]
        BD = self._rel_shift(BD)

        attn = (AC + BD) * self.scale                                # [B,H,T,T]

        # mask handling (bool -> -inf)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                m = torch.zeros_like(attn_mask, dtype=attn.dtype)
                m.masked_fill_(attn_mask, float("-inf"))
                attn_mask = m
            # expected [T,T] or [B,T,T]
            if attn_mask.dim() == 2:
                attn = attn + attn_mask[None, None, :, :]
            elif attn_mask.dim() == 3:
                attn = attn + attn_mask[:, None, :, :]

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum("bhts,sbhd->tbhd", attn, v)              # [T,B,H,D]
        out = out.contiguous().view(T, B, C)                        # [T,B,C]

        if self.batch_first:
            out = out.transpose(0, 1)                               # [B,T,C]

        out = self.out_proj(out)
        out = self.out_drop(out)
        return out


class XL_based_ToMECustomResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        scale_cosine_attn: bool = False,
        scale_heads: bool = False,
        scale_attn: bool = False,
        scale_fc: bool = False,
        batch_first: bool = True,
        tome_r: int = 0,
        tome_class_token: bool = True,
        tome_distill_token: bool = False,
        tome_use_wavg: bool = True,
        tome_metric: str = "x", # "x" or "ln2" (metric to compute similarity)
    ):
        super().__init__()

        # original block parts from ``CustomResidualAttentionBlock``
        self.ln_1 = norm_layer(d_model)
        self.attn = XLAttention(
            d_model,
            n_head,
            qkv_bias=True,
            batch_first=batch_first,
            attn_drop=0.,
            proj_drop=0.,
            max_len=1024,   # set >= max tokens you might have
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()


        # ToMe params
        self.tome_r = int(tome_r)
        self.tome_class_token = tome_class_token
        self.tome_distill_token = tome_distill_token
        self.tome_use_wavg = tome_use_wavg
        assert tome_metric in ("x", "ln2")
        self.tome_metric = tome_metric
        # token size tracker for weighted merges
        self.register_buffer("_tome_size", None, persistent=False)

    def get_reference_weight(self):
        return self.mlp.c_fc.weight

    def _get_size(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        if self._tome_size is None or self._tome_size.shape[:2] != x.shape[:2]:
            self._tome_size = torch.ones(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)
        return self._tome_size

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # Attention residual
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))

        # ToMe merging step
        if self.tome_r > 0:
            metric = x if self.tome_metric == "x" else self.ln_2(x)
            merge, _ = bipartite_soft_matching(
                metric=metric,
                r=self.tome_r,
                class_token=self.tome_class_token,
                distill_token=self.tome_distill_token,
            )

            if self.tome_use_wavg:
                size = self._get_size(x)
                x, size = merge_wavg(merge, x, size=size)
                self._tome_size = size
            else:
                x = merge(x, mode="mean")
                self._tome_size = None

        # MLP residual
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x



class XL_ToMETransformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        batch_first: bool = True,
        tome_r_schedule=0, # ToMe schedule: list length=layers or scalar
        tome_class_token: bool = True,
        tome_distill_token: bool = False,
        tome_use_wavg: bool = True,
        tome_metric: str = "x",
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first
        self.grad_checkpointing = False

        if isinstance(tome_r_schedule, int):
            tome_r_schedule = [tome_r_schedule] * layers
        assert len(tome_r_schedule) == layers

        self.resblocks = nn.ModuleList([
            XL_based_ToMECustomResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                batch_first=batch_first,
                tome_r=int(tome_r_schedule[i]),
                tome_class_token=tome_class_token,
                tome_distill_token=tome_distill_token,
                tome_use_wavg=tome_use_wavg,
                tome_metric=tome_metric,
            )
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()

        for blk in self.resblocks:
            x = blk(x, attn_mask=attn_mask)

        if not self.batch_first:
            x = x.transpose(0, 1)
        return x


class XL_ToMEViT(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        ls_init_value: float = None,
        attentional_pool: bool = False,
        attn_pooler_queries: int = 256,
        attn_pooler_heads: int = 8,
        output_dim: int = 512,
        patch_dropout: float = 0.,
        no_ln_pre: bool = False,
        pos_embed_type: str = 'learnable',
        pool_type: str = 'tok',
        final_ln_after_pool: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        output_tokens: bool = False,

        # ToMe explicit params
        tome_r_schedule: Union[int, List[int]] = 0, # int or list length=layers
        tome_class_token: bool = True,
        tome_distill_token: bool = False,
        tome_use_wavg: bool = True,
        tome_metric: str = "x", # "x" or "ln2"
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        assert tome_metric in ("x", "ln2")

        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim
        self.width = width

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width)
            )
        elif pos_embed_type == 'sin_cos_2d':
            assert self.grid_size[0] == self.grid_size[1], \
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width),
                requires_grad=False
            )
            pe = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pe).float())
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}")

        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()
        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)

        self.transformer = XL_ToMETransformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            batch_first=True,

            tome_r_schedule=tome_r_schedule,
            tome_class_token=tome_class_token,
            tome_distill_token=tome_distill_token,
            tome_use_wavg=tome_use_wavg,
            tome_metric=tome_metric,
        )

        # pooling head (unchanged)
        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim, width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim, width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    raise ValueError(f"Unknown attentional_pool type: {attentional_pool}")
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim, width,
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

        self.init_parameters()

    def init_parameters(self):
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.transformer.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'positional_embedding', 'class_embedding'}

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x
        return pooled, tokens

    def _embeds(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)                 # [B, C, Gh, Gw]
        x = x.reshape(x.shape[0], x.shape[1], -1)   # [B, C, Gh*Gw]
        x = x.permute(0, 2, 1)           # [B, Gh*Gw, C]

        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        return x

    def _pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.attn_pool is not None:
            if getattr(self, "attn_pool_contrastive", None) is not None:
                x = self.ln_post(x)
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    pooled = self.attn_pool_contrastive(tokens)
            else:
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

    def encode(self, x: torch.Tensor):
        x = self._embeds(x)
        x = self.transformer(x)
        return x

    def forward(self, x: torch.Tensor):
        x = self._embeds(x)
        x = self.transformer(x)
        pooled, tokens = self._pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens
        return pooled