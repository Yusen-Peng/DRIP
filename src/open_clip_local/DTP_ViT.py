# Adapted from: 
# https://github.com/PiotrNawrot/dynamic-pooling/blob/main/hourglass.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Tuple, List
import torch

def final(foo,
          upsample):
    """
        Input:
            B x L x S
    """
    autoregressive = foo != 0
    lel = 1 - foo

    lel[autoregressive] = 0

    dim = 2 if upsample else 1

    lel = lel / (lel.sum(dim=dim, keepdim=True) + 1e-9)

    return lel

def common(boundaries, upsample=False):
    boundaries = boundaries.clone()

    n_segments = boundaries.sum(dim=-1).max().item()

    if upsample:
        n_segments += 1

    if n_segments == 0:
        return None

    tmp = torch.zeros_like(
        boundaries
    ).unsqueeze(2) + torch.arange(
        start=0,
        end=n_segments,
        device=boundaries.device
    )

    hh1 = boundaries.cumsum(1)

    if not upsample:
        hh1 -= boundaries

    foo = tmp - hh1.unsqueeze(-1)

    return foo


def downsample(boundaries, hidden, null_group):
    """
        Downsampling

        - The first element of boundaries tensor is always 0 and doesn't matter
        - 1 starts a new group
        - We append an extra "null" group at the beginning
        - We discard last group because it won't be used (in terms of upsampling)

        Input:
            boundaries: B x L
            hidden: L x B x D
        Output:
            shortened_hidden: S x B x D
    """

    foo = common(boundaries, upsample=False)  # B x L x S

    if foo is None:
        return null_group.repeat(1, hidden.size(1), 1)
    else:
        bar = final(foo=foo, upsample=False)  # B x L x S

        shortened_hidden = torch.einsum('lbd,bls->sbd', hidden, bar)
        shortened_hidden = torch.cat(
            [null_group.repeat(1, hidden.size(1), 1), shortened_hidden], dim=0
        )

        return shortened_hidden

def upsample(boundaries, shortened_hidden):
    """
        Upsampling

        - The first element of boundaries tensor is always 0 and doesn't matter
        - 1 starts a new group
        - i-th group can be upsampled only to the tokens from (i+1)-th group, otherwise there's a leak

        Input:
            boundaries: B x L
            shortened_hidden: S x B x D
        Output:
            upsampled_hidden: L x B x D
    """

    foo = common(boundaries, upsample=True)  # B x L x S
    bar = final(foo, upsample=True)  # B x L x S

    return torch.einsum('sbd,bls->lbd', shortened_hidden, bar)

class BoundaryPredictor(nn.Module):
    def __init__(self, d_model, d_inner, activation_function,
                 temp, prior, bp_type, threshold=0.5,
                 image_size=None, patch_size=None, embed_dim=None):
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.bp_type = bp_type
        self.threshold = threshold
        self.compression_rate = prior
        self.embed_dim = embed_dim
        if image_size is not None and patch_size is not None:
            self.image_size = image_size
            self.patch_size = patch_size
            self.num_patches = (image_size // patch_size) ** 2

        if activation_function == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == 'gelu':
            activation_fn = torch.nn.GELU()

        self.boundary_predictor = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Linear(d_inner, 1),
        )

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, hidden):
        # Hidden is of shape [seq_len x bs x d_model]
        # Boundaries we return are [bs x seq_len]
        boundary_logits = self.boundary_predictor(hidden).squeeze(-1).transpose(0, 1)
        boundary_probs = torch.sigmoid(boundary_logits)

        if self.bp_type == 'gumbel':
            bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=self.temp,
                probs=boundary_probs,
            )

            soft_boundaries = bernoulli.rsample()

            hard_boundaries = (soft_boundaries > self.threshold).float()
            hard_boundaries = (
                hard_boundaries - soft_boundaries.detach() + soft_boundaries
            )
        elif self.bp_type in ['entropy', 'unigram']:
            soft_boundaries = boundary_probs
            hard_boundaries = (soft_boundaries > self.threshold).float()

        return soft_boundaries, hard_boundaries

    def calc_loss(self, preds, gt):
        # B x T
        if self.bp_type in ['entropy', 'unigram']:
            assert preds is not None and gt is not None
            return self.loss(preds, gt.float())
        elif self.bp_type in ['gumbel']:
            assert gt is None
            binomial = torch.distributions.binomial.Binomial(
                preds.size(-1),
                probs=torch.Tensor([self.prior]).to(preds.device)
            )
            loss_boundaries = -binomial.log_prob(
                preds.sum(dim=-1)
            ).mean() / preds.size(-1)

            return loss_boundaries

    def calc_stats(self, preds, gt):
        # B x T
        preds, gt = preds.bool(), gt.bool()
        TP = ((preds == gt) & preds).sum().item()
        FP = ((preds != gt) & preds).sum().item()
        FN = ((preds != gt) & (~preds)).sum().item()

        acc = (preds == gt).sum().item() / gt.numel()

        if TP == 0:
            precision, recall = 0, 0
        else:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

        stats = {
            'acc': acc,
            'precision': precision,
            'recall': recall
        }

        return stats

class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_chans: int = 3, embed_dim: int = 768):
        """
        Patch Embedding Layer
        Args:
            image_size (int): Size of the input image (assumed square).
            patch_size (int): Size of each patch (assumed square).
            in_chans (int): Number of input channels (e.g., 3 for RGB).
            embed_dim (int): Dimension of the embedding space.
        """
        super().__init__()
        self.img_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size // patch_size, image_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        """
            input: [batch size, # channels, height, width]
            output: [batch size, # patches, embed_dim]
        """
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class TransformerBlock(nn.Module):
    """
    A single Transformer block with multi-head self-attention and MLP layers.
    Args:
        dim (int): Dimension of the input features.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
        drop (float): Dropout rate applied after attention and MLP layers.
        attn_drop (float): Dropout rate applied within attention layers.
        activation_function (str): Activation function used in MLP layers ('gelu' or 'relu').
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0., attn_drop=0., activation_function='gelu'):
        
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=False)
        self.drop_path = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)

        act_fn = nn.GELU() if activation_function == 'gelu' else nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            act_fn,
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DTPViT(nn.Module):
    """
    DTP-ViT: A Vision Transformer with Dynamic Token Pooling
    Args:
        image_size (int): Size of the input image (assumed square).
        patch_size (int): Size of each patch (assumed square).
        in_chans (int): Number of input channels (e.g., 3 for RGB).
        embed_dim (int): Dimension of the embedding space.
        depth (Tuple[int]): Number of transformer blocks in each stage.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
        drop_rate (float): Dropout rate applied after positional embedding and within transformer blocks.
        attn_drop_rate (float): Dropout rate applied within attention layers.
        temp (float): Temperature for the Gumbel-Sigmoid boundary predictor.
        compression_rate (float): Compression rate for token pooling.
        threshold (float): Cutoff used to decide whether a patch token should be kept or dropped.
        activation_function (str): Activation function used in MLP layers.
        num_classes (int): Number of output classes for classification tasks.
    """
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: Tuple = (2, 8, 2), # (Pre, Shorten, Post), (2, 8, 2) from the offical paper
        num_heads: int = 12,
        mlp_ratio: float = 4.0,   # the size of the MLP (feedforward) layer relative to embed_dim
        drop_rate: float = 0.1,   # dropout rate applied (1) after positional embedding and within transformer blocks
        attn_drop_rate: float = 0.1, # dropout rate applied within attention layers
        temp: float = 0.5, # temperature for the Gumbel-Sigmoid boundary predictor
        compression_rate: float = 0.1,
            # 0.5: 2x compression
            # 0.25: 4x compression
            # 0.1: 10x compression
        threshold: float = 0.5,   # the cutoff used to decide whether a patch token should be kept or dropped
        activation_function: str = 'gelu',
        num_classes: int = 1000,
        flop_measure: bool = False,  # whether to measure FLOPs
    ):
        super(DTPViT, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth  # (Pre, Shorten, Post)
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.temp = temp
        self.threshold = threshold
        self.num_classes = num_classes
        self.activation_function = activation_function
        # whether to measure FLOPs
        # if True, we will simulate the boundaries to satisfy the compression rate
        # if False, we will use the BoundaryPredictor to predict the boundaries
        self.flop_measure = flop_measure
        self.compression_rate = compression_rate


        # patch embeddings
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_chans, embed_dim)

        # positional embeddings

        # === CLS token ===
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # [1, 1, D]
        nn.init.normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)


        # dropout after positional embedding
        self.pos_drop = nn.Dropout(p=drop_rate)

        # transformer blocks
        self.pre_blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate, activation_function)
            for _ in range(depth[0])
        ])
        self.shorten_blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate, activation_function)
            for _ in range(depth[1])
        ])


        # NOTE: no upsampling in DTP-ViT, so no post blocks!
        # self.post_blocks = nn.Sequential(*[
        #     TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate, activation_function)
        #     for _ in range(depth[2])
        # ])

        self.boundary_predictor = BoundaryPredictor(
            d_model=embed_dim,
            d_inner=embed_dim * 2,
            activation_function=activation_function,
            temp=temp,
            prior=compression_rate,
            bp_type="gumbel",
            threshold=threshold,
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.null_group = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.null_group)

        print("=" * 70)
        print("[INFO] Congratulations! You have successfully initialized DTP-ViT!")
        print(f"Compression Rate: {compression_rate}")
        print(f"depth of each section: {depth}")
        print("=" * 70)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, 3, H, W]
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, 1 + N, D]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = x.transpose(0, 1)

        x = self.pre_blocks(x)

        # NOTE: remove residual connection since we are not upsampling anymore
        # residual = x

        # NOTE: we simulate the boundaries, like DynamicViT folks did.
        # https://github.com/raoyongming/DynamicViT/blob/master/models/dylvvit.py
        if self.flop_measure:
            print("=" * 80)
            print("[INFO] FLOP measurement mode: simulating fake boundaries to satisfy the compression rate.")
            print("=" * 80)
            L = x.shape[0]
            interval = max(1, int(1 / self.compression_rate))
            hard_boundaries = torch.zeros(B, L, device=x.device)
            hard_boundaries[:, ::interval] = 1
        else:
            _, hard_boundaries = self.boundary_predictor(x)

        x = downsample(hard_boundaries, x, self.null_group)
        x = self.norm(x)


        x = self.shorten_blocks(x)

        # NOTE: no upsampling!
        # x = upsample(hard_boundaries, x)
        
        # NOTE: again, remove residual connection - no upsampling
        #x = x + residual
        
        # x = self.post_blocks(x) 

        x = x.transpose(0, 1)
        x = self.norm(x)

        x = x.mean(dim=1)
        return self.head(x)


###########################################################################
# =================== above is the DTP-ViT ===================
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