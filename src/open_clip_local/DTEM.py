"""
    Code adapted from DTEM.
    https://github.com/movinghoon/DTEM/blob/main/dtem/dtem.py
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import VisionTransformer
from timm.models.vision_transformer import Attention, Block
from functools import partial
from typing import Callable, Tuple


def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
) -> Tuple[Callable, Callable]:
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


class DTEMLinear(nn.Linear):
    def __init__(self, qkv_layer, feat_dim):
        super().__init__(in_features=qkv_layer.weight.shape[1], out_features=qkv_layer.weight.shape[0] + feat_dim, bias=True)
        # qkv
        self.qkv_layer = qkv_layer

        # metric
        self.feat_dim = feat_dim
        self.metric_layer = nn.Linear(qkv_layer.weight.shape[-1], feat_dim)

        # copy
        self.update()

    @torch.no_grad()
    def update(self):
        # qkv -> self
        self.weight[:-self.feat_dim].copy_(self.qkv_layer.weight)
        self.bias[:-self.feat_dim].copy_(self.qkv_layer.bias)
        
        # metric_layer -> self
        self.weight[-self.feat_dim:].copy_(self.metric_layer.weight)
        self.bias[-self.feat_dim:].copy_(self.metric_layer.bias)

    def train(self, mode=True):
        if mode is False:   # if eval
            self.update()
        return super().train(mode)

    def forward(self, input: torch.Tensor):
        if not self.training:
            out = F.linear(input, self.weight, self.bias)
            return out[..., :-self.feat_dim], out[..., -self.feat_dim:]
        
        # training
        out1 = self.qkv_layer(input)
        out2 = self.metric_layer(input.detach())
        return out1, out2


"""
    timm - deit patch
"""
class DTEMAttention(Attention):
    def patch(self, feat_dim=None):
        if feat_dim is not None:
            out_dim = feat_dim
        else:
            dim = self.head_dim * self.num_heads
            out_dim = self.head_dim if dim < 1024 else 2 * self.head_dim
        
        # add metric_layer
        self.qkv = DTEMLinear(self.qkv, out_dim)
    
    def forward(self, x, size=None, prop_attn=True):    # x:(B, N, C), size:(B, N)
        B, N, C = x.shape
        out1, out2 = self.qkv(x)
        qkv = out1.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)     # B, H, N, head_dim
        q, k = self.q_norm(q), self.k_norm(k)

        # fp32 for softmax computation
        q, k, v = q.type(torch.float32), k.type(torch.float32), v.type(torch.float32)
        with torch.cuda.amp.autocast(dtype=torch.float32, enabled=True):
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            
            if size is None or (not prop_attn): # for MAE
                attn = attn.softmax(dim=-1)
            else:   # as in DynamicViT
                _attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]
                _attn = _attn.exp_() * size[:, None, None, :].type(torch.float32)
                attn = _attn / _attn.sum(dim=-1, keepdim=True)
            attn = self.attn_drop(attn)
            _x = attn @ v
        x = _x.type(x.dtype)
        
        # output
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # out_dict
        out_dict = {
            'q': q,
            'k': k,
            'v': v,
            'x': x,
            'metric': out2}
        return x, out_dict


class DTEMBlock(Block):
    def patch(self, k2, tau1, tau2, feat_dim=None):
        # add topk operator
        self.k2 = k2
        self.tau1 = tau1
        self.tau2 = tau2
        
        # patch attention
        self.attn.__class__ = DTEMAttention
        self.attn.patch(feat_dim=feat_dim)
    
    def _select(self, x, k):
        EPSILON = torch.finfo(torch.float32).eps   # 1.1920928955078125e-07
        
        # select
        x = x.type(torch.float32)
        with torch.cuda.amp.autocast(dtype=torch.float32, enabled=True):
            # mask
            _idx = x.argsort(dim=-1, descending=True)[..., :self.k2]
            _x = x.gather(dim=-1, index=_idx)
            
            # scale
            _x = _x / self.tau1
            
            # group
            B, N, M = _x.shape
            khot = torch.zeros_like(_x)
            for _ in range(k):
                onehot_approx = F.softmax(_x.view(B, -1) / self.tau2, dim=-1).view(B, N, M)
                khot += onehot_approx
                khot_mask = torch.clamp(1 - onehot_approx.sum(dim=-1, keepdim=True), min=EPSILON)
                _x = _x + torch.log(khot_mask)
        
        # new 
        tmp = torch.clamp(khot.sum(dim=-1, keepdim=True).detach() - 1, min=0.) + 1.
        nkhot = khot / tmp
        
        # scatter
        assign = torch.zeros_like(x).scatter_reduce(-1, _idx, nkhot, reduce='sum')
        
        # out_dict
        with torch.no_grad():
            out_dict = {
                'num': nkhot.sum().item(),
                'max': khot.view(B, -1).max(dim=-1)[0].sum().item(),
            }
        return assign, out_dict

    def _merge_train(self, x, size, r, n, out_dict):
        # metric
        metric = out_dict['metric']
        metric = metric / metric.norm(dim=-1, keepdim=True)
        
        # merge profile
        n = n if self.training else x.size()[1]
        r = min(r, (n - 1) // 2)    # accounts for CLS token
        
        # split - only n tokens participates
        xa, xb = x[..., 1:n:2, :], x[..., 2:n:2, :]
        a, b = metric[..., 1:n:2, :], metric[..., 2:n:2, :]
        wa, wb = size[..., 1:n:2], size[..., 2:n:2]
        
        # scores divided by temperature
        scores = a @ b.transpose(-1, -2)
        
        # select
        assign, _out = self._select(scores, k=r)
        
        # merge operation
        xb = wb[..., None] * xb + assign.transpose(-1, -2) @ (wa[..., None] * xa)   # patch update - 1
        wb = wb + (assign.transpose(-1, -2) @ wa[..., None])[..., 0]    # effective size update on wb
        tmp = 1 - assign.sum(dim=-1)    # for clip
        wa = wa * (tmp + (torch.clamp(tmp, min=0., max=1.) - tmp).detach())     # numerical stability -- sometimes tmp < 0 happen...?
        xb = xb / wb[..., None]     # patch update - 2
        
        # concat first
        w = torch.cat([wa, wb], dim=-1)
        nx = torch.cat([xa, xb], dim=1)
        
        # sorted idxs
        nidxs = w.argsort(dim=-1, descending=True)
        
        # sort nx and w
        w = w.gather(dim=-1, index=nidxs)
        nx = nx.gather(dim=-2, index=nidxs[..., None].expand_as(nx))

        # output
        x_output = torch.cat([x[:, :1], nx, x[:, n:]], dim=1)
        size_output = torch.cat([size[:, :1], w, size[:, n:]], dim=-1)
        return x_output, size_output, n - r, _out

    def _merge_eval(self, x, size, r, out_dict):    # the same to ToMe
        metric = out_dict['metric']
        metric = metric / metric.norm(dim=-1, keepdim=True)

        merge, _ = bipartite_soft_matching(metric, r=r, class_token=True)
        x = merge(x * size[..., None], mode='sum')
        size = merge(size[..., None], mode='sum')
        x = x / size
        return x, size[..., 0], x.size(1), None

    def merge(self, x, size, r, n, out_dict):
        return self._merge_train(x, size, r, n, out_dict) if self.training else self._merge_eval(x, size, r, out_dict)
    
    def forward(self, x, size=None, r=None, n=None, prop_attn=True):
        # Attn
        tmp, out_dict = self.attn(self.norm1(x), size=size, prop_attn=prop_attn)
        x = x + self.drop_path1(self.ls1(tmp))
        
        # Merging
        if size is not None and r > 0 and n > 0:
            x, size, n, out_dict = self.merge(x, size, r, n, out_dict)

        # FFN
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, size, n, out_dict


class DTEM(VisionTransformer):
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        
        # blocks
        n = x.size(1)
        out_dicts = []
        size = torch.ones_like(x[..., 0])
        r = self.r if isinstance(self.r, list) else [self.r for _ in range(len(self.blocks))]
        for i, block in enumerate(self.blocks):
            x, size, n, out_dict = block(x, size, r[i], n)
            out_dicts.append(out_dict)
        x = self.norm(x)
        return x, out_dicts
    
    def forward(self, x, return_out_dicts=False):
        x, out_dicts = self.forward_features(x)
        x = self.forward_head(x)
        if return_out_dicts:
            return x, out_dicts
        return x
    
    def update_r(self, r):
        self.r = r

    def patch(self, k2, tau1, tau2, feat_dim):
        self.r = 0
        for block in self.blocks:
            block.__class__ = DTEMBlock
            block.patch(k2, tau1, tau2, feat_dim)
            

def patch_deit(model: VisionTransformer, k2=3, tau1=0.1, tau2=0.1, feat_dim=None, **kwargs) -> DTEM:
    model.__class__ = DTEM
    model.patch(k2, tau1, tau2, feat_dim)
    return model


"""
    timm - mae patch
"""
class MAEDTEM(VisionTransformer):
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        
        # blocks
        n = x.size(1)
        size = torch.ones_like(x[..., 0])
        r = self.r if isinstance(self.r, list) else [self.r for _ in range(len(self.blocks))]
        out_dicts = []
        for i, block in enumerate(self.blocks):
            x, size, n, out_dict = block(x, size, r[i], n, prop_attn=True if self.training else False)
            out_dicts.append(out_dict)
        
        if self.global_pool:
            # Wheter prop_pool or not
            x = (x[:, 1:] * size[..., None][:, 1:]).sum(dim=1) / size[..., None][:, 1:].sum(dim=1) if self.prop_pool else x[:, 1:n, :].mean(dim=1) 
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
            
        return outcome, out_dicts
    
    def update_r(self, r):
        self.r = r

    def patch(self, k2, tau1, tau2, prop_pool=False, feat_dim=None):
        self.r = 0
        self.prop_pool = prop_pool
        for block in self.blocks:
            block.__class__ = DTEMBlock
            block.patch(k2, tau1, tau2, feat_dim)

    def forward(self, x, return_out_dicts=False):
        x, out_dicts = self.forward_features(x)
        x = self.head(x)
        if return_out_dicts:
            return x, out_dicts
        return x


def patch_mae(model: VisionTransformer, k2=3, tau1=0.1, tau2=0.1, feat_dim=None, prop_pool=True, **kwargs):
    model.__class__ = MAEDTEM
    model.patch(k2, tau1, tau2, prop_pool, feat_dim)
    return model


def patch(model: VisionTransformer, k2, tau1, tau2, feat_dim=None, mae=False, prop_pool=False):
    _patch = partial(patch_mae, prop_pool=prop_pool) if mae else patch_deit
    model = _patch(model, k2, tau1, tau2, feat_dim)
    return model



class LogitsOnly(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        # Always return a Tensor for fvcore
        out = self.model(x, return_out_dicts=False) if "return_out_dicts" in self.model.forward.__code__.co_varnames else self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

class DTEMXLAttention(Attention):
    def patch(self, feat_dim=None, max_len=1024):
        if feat_dim is not None:
            out_dim = feat_dim
        else:
            dim = self.head_dim * self.num_heads
            out_dim = self.head_dim if dim < 1024 else 2 * self.head_dim

        self.qkv = DTEMLinear(self.qkv, out_dim)

        self.max_len = max_len
        self.rel_emb = nn.Embedding(2 * max_len - 1, self.num_heads * self.head_dim)
        nn.init.normal_(self.rel_emb.weight, mean=0.0, std=0.02)

        self.r_net = nn.Linear(self.num_heads * self.head_dim, self.num_heads * self.head_dim, bias=False)
        self.r_w_bias = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.r_r_bias = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        # x: [B,H,N,N]
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3)                      # [B,H,N,N+1]
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))  # [B,H,N+1,N]
        x = x_padded[:, :, 1:, :].view_as(x)                            # [B,H,N,N]
        return x

    def _build_r(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if n > self.max_len:
            raise ValueError(f"n={n} > max_len={self.max_len} (increase max_len)")
        dist = torch.arange(n - 1, -1, -1, device=device)               # [n]
        idx = (self.max_len - 1) + dist                                 # shift
        r = self.rel_emb(idx).to(dtype=dtype)                           # [n, H*D]
        r = self.r_net(r)                                               # [n, H*D]
        r = r.view(n, self.num_heads, self.head_dim)                    # [n,H,D]
        return r

    def forward(self, x: torch.Tensor, size=None, prop_attn=True):
        B, N, C = x.shape

        out1, out2 = self.qkv(x)
        qkv = out1.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B,H,N,D]
        q, k = self.q_norm(q), self.k_norm(k)

        # fp32 compute for stability (same style you used)
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)

        # build relative keys r: [N,H,D]
        r = self._build_r(N, x.device, q.dtype)

        with torch.amp.autocast(dtype=torch.float32, enabled=True, device_type='cuda'):
            # AC term
            rw_q = q + self.r_w_bias[None, :, None, :]                      # [B,H,N,D]
            AC = torch.einsum("bhid,bhjd->bhij", rw_q, k)                   # [B,H,N,N]

            # BD term
            rr_q = q + self.r_r_bias[None, :, None, :]
            BD = torch.einsum("bhid,jhd->bhij", rr_q, r)                    # [B,H,N,N]
            BD = self._rel_shift(BD)

            attn = (AC + BD) * self.scale

            if size is None or (not prop_attn):
                attn = attn.softmax(dim=-1)
            else:
                # same prop_attn logic as DTEMAttention
                _attn = attn - torch.max(attn, dim=-1, keepdim=True)[0]
                _attn = _attn.exp_() * size[:, None, None, :].to(torch.float32)
                attn = _attn / _attn.sum(dim=-1, keepdim=True)

            attn = self.attn_drop(attn)
            _x = attn @ v  # [B,H,N,D]

        x = _x.to(x.dtype).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        out_dict = {
            "x": x,
            "metric": out2,
            # optional debugging:
            # "attn": attn,
        }
        return x, out_dict


class XL_DTEMBlock(Block):
    def patch(self, k2, tau1, tau2, feat_dim=None):
        # add topk operator
        self.k2 = k2
        self.tau1 = tau1
        self.tau2 = tau2
        
        # patch attention
        self.attn.__class__ = DTEMXLAttention
        self.attn.patch(feat_dim=feat_dim)
    
    def _select(self, x, k):
        EPSILON = torch.finfo(torch.float32).eps   # 1.1920928955078125e-07
        
        # select
        x = x.type(torch.float32)
        with torch.cuda.amp.autocast(dtype=torch.float32, enabled=True):
            # mask
            _idx = x.argsort(dim=-1, descending=True)[..., :self.k2]
            _x = x.gather(dim=-1, index=_idx)
            
            # scale
            _x = _x / self.tau1
            
            # group
            B, N, M = _x.shape
            khot = torch.zeros_like(_x)
            for _ in range(k):
                onehot_approx = F.softmax(_x.view(B, -1) / self.tau2, dim=-1).view(B, N, M)
                khot += onehot_approx
                khot_mask = torch.clamp(1 - onehot_approx.sum(dim=-1, keepdim=True), min=EPSILON)
                _x = _x + torch.log(khot_mask)
        
        # new 
        tmp = torch.clamp(khot.sum(dim=-1, keepdim=True).detach() - 1, min=0.) + 1.
        nkhot = khot / tmp
        
        # scatter
        assign = torch.zeros_like(x).scatter_reduce(-1, _idx, nkhot, reduce='sum')
        
        # out_dict
        with torch.no_grad():
            out_dict = {
                'num': nkhot.sum().item(),
                'max': khot.view(B, -1).max(dim=-1)[0].sum().item(),
            }
        return assign, out_dict

    def _merge_train(self, x, size, r, n, out_dict):
        # metric
        metric = out_dict['metric']
        metric = metric / metric.norm(dim=-1, keepdim=True)
        
        # merge profile
        n = n if self.training else x.size()[1]
        r = min(r, (n - 1) // 2)    # accounts for CLS token
        
        # split - only n tokens participates
        xa, xb = x[..., 1:n:2, :], x[..., 2:n:2, :]
        a, b = metric[..., 1:n:2, :], metric[..., 2:n:2, :]
        wa, wb = size[..., 1:n:2], size[..., 2:n:2]
        
        # scores divided by temperature
        scores = a @ b.transpose(-1, -2)
        
        # select
        assign, _out = self._select(scores, k=r)
        
        # merge operation
        xb = wb[..., None] * xb + assign.transpose(-1, -2) @ (wa[..., None] * xa)   # patch update - 1
        wb = wb + (assign.transpose(-1, -2) @ wa[..., None])[..., 0]    # effective size update on wb
        tmp = 1 - assign.sum(dim=-1)    # for clip
        wa = wa * (tmp + (torch.clamp(tmp, min=0., max=1.) - tmp).detach())     # numerical stability -- sometimes tmp < 0 happen...?
        xb = xb / wb[..., None]     # patch update - 2
        
        # concat first
        w = torch.cat([wa, wb], dim=-1)
        nx = torch.cat([xa, xb], dim=1)
        
        # sorted idxs
        nidxs = w.argsort(dim=-1, descending=True)
        
        # sort nx and w
        w = w.gather(dim=-1, index=nidxs)
        nx = nx.gather(dim=-2, index=nidxs[..., None].expand_as(nx))

        # output
        x_output = torch.cat([x[:, :1], nx, x[:, n:]], dim=1)
        size_output = torch.cat([size[:, :1], w, size[:, n:]], dim=-1)
        return x_output, size_output, n - r, _out

    def _merge_eval(self, x, size, r, out_dict):    # the same to ToMe
        metric = out_dict['metric']
        metric = metric / metric.norm(dim=-1, keepdim=True)

        merge, _ = bipartite_soft_matching(metric, r=r, class_token=True)
        x = merge(x * size[..., None], mode='sum')
        size = merge(size[..., None], mode='sum')
        x = x / size
        return x, size[..., 0], x.size(1), None

    def merge(self, x, size, r, n, out_dict):
        return self._merge_train(x, size, r, n, out_dict) if self.training else self._merge_eval(x, size, r, out_dict)
    
    def forward(self, x, size=None, r=None, n=None, prop_attn=True):
        # Attn
        tmp, out_dict = self.attn(self.norm1(x), size=size, prop_attn=prop_attn)
        x = x + self.drop_path1(self.ls1(tmp))
        
        # Merging
        if size is not None and r > 0 and n > 0:
            x, size, n, out_dict = self.merge(x, size, r, n, out_dict)

        # FFN
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x, size, n, out_dict

class XL_DTEM(VisionTransformer):
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        
        # blocks
        n = x.size(1)
        out_dicts = []
        size = torch.ones_like(x[..., 0])
        r = self.r if isinstance(self.r, list) else [self.r for _ in range(len(self.blocks))]
        for i, block in enumerate(self.blocks):
            x, size, n, out_dict = block(x, size, r[i], n)
            out_dicts.append(out_dict)
        x = self.norm(x)
        return x, out_dicts
    
    def forward(self, x, return_out_dicts=False):
        x, out_dicts = self.forward_features(x)
        x = self.forward_head(x)
        if return_out_dicts:
            return x, out_dicts
        return x
    
    def update_r(self, r):
        self.r = r

    def patch(self, k2, tau1, tau2, feat_dim):
        self.r = 0
        for block in self.blocks:
            block.__class__ = XL_DTEMBlock
            block.patch(k2, tau1, tau2, feat_dim)

def patch_deit_XL(model: VisionTransformer, k2=3, tau1=0.1, tau2=0.1, feat_dim=None, **kwargs) -> XL_DTEM:
    model.__class__ = XL_DTEM
    model.patch(k2, tau1, tau2, feat_dim)
    return model
