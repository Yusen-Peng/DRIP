
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Tuple, List
import torch
from DTP_ViT import BoundaryPredictor, downsample
# =================== below is the DTP-ViT_XL ===================
# Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
###########################################################################

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]

@torch.jit.script
def add_and_scale(tensor1, tensor2, alpha: float):
    return alpha * (tensor1 + tensor2)

class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(
        self, n_head, d_model, d_head, dropout, dropatt, pre_lnorm, activation_function
    ):
        super(RelPartialLearnableMultiHeadAttn, self).__init__()

        del activation_function

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(self.d_model, 3 * n_head * d_head)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3)

        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))

        x = x_padded.narrow(2, 1, x_padded.size(2) - 1).view_as(x)

        return x

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask):
        # w is of size: T x B x C
        # r is of size: T x 1 x C
        # biases are of size: (n_head x d_head), we add the same bias to each token
        # attn_mask is of size (q_len x k_len)
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if self.pre_lnorm:
            w_head_q, w_head_k, w_head_v = self.qkv_net(self.layer_norm(w))
        else:
            w_heads = self.qkv_net(w)

        r_head_k = self.r_net(r)
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)       # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + r_w_bias                                # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->bnij', rw_head_q, w_head_k)      # bsz x n_head x qlen x klen

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->bnij', rr_head_q, r_head_k)       # bsz x n_head x qlen x klen
        BD = self._rel_shift(BD)

        # [bsz x n_head x qlen x klen]
        attn_score = add_and_scale(AC, BD, self.scale)

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, None, :, :], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, None, :, :], -float('inf'))
        else:
            pass
        
        # [bsz x n_head x qlen x klen]
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('bnij,jbnd->ibnd', attn_prob, w_head_v)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm, activation_function):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        if activation_function == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == 'gelu':
            activation_fn = torch.nn.GELU()

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            core_out = self.CoreNet(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp)
            output = self.layer_norm(inp + core_out)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        dropatt,
        pre_lnorm,
        activation_function,
    ):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, dropatt, pre_lnorm, activation_function
        )
        self.pos_ff = PositionwiseFF(
            d_model,
            d_inner,
            dropout,
            pre_lnorm,
            activation_function,
        )

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask)
        output = self.pos_ff(output)

        return output

class DTPViT_XL(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=(2, 8, 0),
                 num_heads=12,
                 mlp_ratio=4.0,
                 drop_rate=0.1,
                 attn_drop_rate=0.1,
                 temp=1.0,
                 prior=0.2,
                 bp_type='gumbel',
                 threshold=0.5,
                 num_classes=1000,
                 flop_measure: bool = False,  # whether to measure FLOPs
        ):

        super().__init__()
        self.flop_measure = flop_measure
        self.prior = prior
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.seq_len = self.num_patches

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.dropout = nn.Dropout(drop_rate)

        self.pos_emb = PositionalEmbedding(embed_dim)
        self.r_w_bias = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))
        self.r_r_bias = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))

        self.pre_blocks = nn.ModuleList([
            RelPartialLearnableDecoderLayer(
                n_head=num_heads,
                d_model=embed_dim,
                d_head=embed_dim // num_heads,
                d_inner=int(embed_dim * mlp_ratio),
                dropout=drop_rate,
                dropatt=attn_drop_rate,
                pre_lnorm=False,
                activation_function='gelu'
            ) for _ in range(depth[0])
        ])

        self.boundary_predictor = BoundaryPredictor(
            d_model=embed_dim,
            d_inner=int(embed_dim * mlp_ratio),
            activation_function='gelu',
            temp=temp,
            prior=prior,
            bp_type=bp_type,
            threshold=threshold
        )

        self.short_blocks = nn.ModuleList([
            RelPartialLearnableDecoderLayer(
                n_head=num_heads,
                d_model=embed_dim,
                d_head=embed_dim // num_heads,
                d_inner=int(embed_dim * mlp_ratio),
                dropout=drop_rate,
                dropatt=attn_drop_rate,
                pre_lnorm=False,
                activation_function='gelu'
            ) for _ in range(depth[1])
        ])

        self.down_ln = nn.LayerNorm(embed_dim)
        self.null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.null_token, std=0.02)

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)  # (B, C, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.dropout(x)

        x = x.transpose(0, 1)  # (T, B, D)

        T = x.size(0)
        pos_seq = torch.arange(T - 1, -1, -1.0, device=x.device, dtype=x.dtype)
        pos_emb = self.pos_emb(pos_seq)
        pos_emb = self.dropout(pos_emb)

        for block in self.pre_blocks:
            x = block(x, pos_emb, self.r_w_bias, self.r_r_bias)

        # NOTE: we simulate the boundaries, like DynamicViT folks did.
        # https://github.com/raoyongming/DynamicViT/blob/master/models/dylvvit.py
        if self.flop_measure:
            print("=" * 80)
            print("[INFO] FLOP measurement mode: simulating fake boundaries to satisfy the compression rate.")
            print("=" * 80)
            L = x.shape[0]
            interval = max(1, int(1 / self.prior))
            hard_boundaries = torch.zeros(B, L, device=x.device)
            hard_boundaries[:, ::interval] = 1
        else:
            _, hard_boundaries = self.boundary_predictor(x)  # B x T

        shortened = downsample(hard_boundaries, x, self.null_token)  # S x B x D

        shortened = self.down_ln(shortened)

        S = shortened.size(0)
        pos_seq_short = torch.arange(S - 1, -1, -1.0, device=x.device, dtype=x.dtype)
        pos_emb_short = self.pos_emb(pos_seq_short)
        pos_emb_short = self.dropout(pos_emb_short)

        x = shortened
        for block in self.short_blocks:
            x = block(x, pos_emb_short, self.r_w_bias, self.r_r_bias)

        x = x.mean(dim=0)
        return self.head(x)