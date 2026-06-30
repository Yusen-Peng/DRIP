"""
    Largely adopted from Perceiver GitHub:
    https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py

"""

from __future__ import annotations
from math import pi, log
from functools import wraps
import torch
from torch import nn, einsum, stack, cat
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Reduce


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l1norm(t, dim = -1, eps = 1e-8):
    return F.normalize(t, p = 1, dim = dim, eps = eps)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)



class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)
    
    def forward(self, x, context = None, mask = None, inverted_attention = False):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h = h) for t in (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        if inverted_attention:
            attn = sim.softmax(dim = -2)

            if exists(mask):
                attn = attn.masked_fill(~mask, 0.)

            attn = l1norm(attn)
        else:
            attn = sim.softmax(dim = -1)

        attn = self.dropout(attn)

        # aggregate

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

class GEGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim,
        num_latents,
        depth=1,
        heads=8,
        dim_head=64,
        ff_mult=4,
        dropout=0.0,
    ):
        super().__init__()

        # learnable query tokens
        # FIXME: randomly initialized for now
        self.latents = nn.Parameter(torch.randn(num_latents, dim))


        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(
                    dim,
                    Attention(
                        query_dim=dim,
                        context_dim=dim,
                        heads=heads,
                        dim_head=dim_head,
                        dropout=dropout,
                    ),
                    context_dim=dim,
                ),
                PreNorm(dim, FeedForward(dim, mult=ff_mult, dropout=dropout)),
                PreNorm(
                    dim,
                    Attention(
                        query_dim=dim,
                        heads=heads,
                        dim_head=dim_head,
                        dropout=dropout,
                    ),
                ),
                PreNorm(dim, FeedForward(dim, mult=ff_mult, dropout=dropout)),
            ])
            for _ in range(depth)
        ])

    def forward(self, patch_tokens):
        # patch_tokens: [B, L, D]
        B = patch_tokens.shape[0]

        x = repeat(self.latents, "n d -> b n d", b=B)

        for cross_attn, cross_ff, self_attn, self_ff in self.layers:
            x = cross_attn(x, context=patch_tokens) + x
            x = cross_ff(x) + x
            x = self_attn(x) + x
            x = self_ff(x) + x

        return x


