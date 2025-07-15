# Adapted from: 
# https://github.com/PiotrNawrot/dynamic-pooling/blob/main/hourglass.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Tuple, List
import torch
import numpy as np

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

        bar = bar.to(hidden.dtype)  # ensure same dtype

        shortened_hidden = torch.einsum('lbd,bls->sbd', hidden, bar)
        shortened_hidden = torch.cat(
            [null_group.repeat(1, hidden.size(1), 1), shortened_hidden], dim=0
        )

        return shortened_hidden

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
        """
            This function is 100% copied and pasted without any changes
            from https://github.com/PiotrNawrot/dynamic-pooling/blob/main/hourglass.py
        """
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
        """
            This function is 100% copied and pasted without any changes
            from https://github.com/PiotrNawrot/dynamic-pooling/blob/main/hourglass.py
        """
        # B x T
        if self.bp_type in ['entropy', 'unigram']:
            assert preds is not None and gt is not None
            return self.loss(preds, gt.float())
        elif self.bp_type in ['gumbel']:
            assert gt is None
            total_count = preds.size(-1)
            target_count = preds.sum(dim=-1)
            binomial = torch.distributions.binomial.Binomial(
                total_count=total_count,
                probs=torch.Tensor([self.prior]).to(preds.device)
            )
            loss_boundaries = -binomial.log_prob(target_count).mean() / total_count
            return loss_boundaries
    
    def calc_loss_lower_bound(self, preds, lambda_val=1.0):
        """
        Compute lower-bound-aware boundary loss:
        loss = max(k/N - beta, 0), where beta = alpha - lambda * sigma

        Args:
            preds: Tensor of shape [B, T] representing boundary probabilities
            lambda_val: lambda hyperparameter (default = 1.0)

        Returns:
            Scalar tensor: computed loss
        """        

        # compute k/N for each sample (boundary rate)
        k_over_N = preds.mean(dim=-1)
        
        # compute standard deviation in this batch
        sigma = k_over_N.std(unbiased=False)

        # print("💀" * 20)
        # print(f"{preds=}")
        # print(f"{k_over_N=}, {sigma=}")

        # calculate dynamic lower bound
        beta = self.prior - lambda_val * sigma

        # compute final loss: max(k/N - beta, 0)
        loss = torch.relu(k_over_N - beta)

        return loss.mean()


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
        lower_bound: bool = False,  # whether to use lower-bound-aware boundary loss
        lambda_val: float = 1.0,  # lambda hyperparameter for dynamic lower bound
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
        self.lower_bound = lower_bound
        self.lambda_val = lambda_val
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

        # CLS token
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
        #print(f"depth of each section: {depth}")
        print("=" * 70)


    def forward(self, x: torch.Tensor, return_loss=False):
        # input shape: [B, 3, H, W]
        B = x.size(0)

        x = self.patch_embed(x)                             # [B, N, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)       # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)               # [B, 1 + N, D]
        x = x + self.pos_embed                              # [B, 1 + N, D]
        x = self.pos_drop(x)                                # [B, 1 + N, D]
        x = x.transpose(0, 1)                               # [1 + N, B, D]

        x = self.pre_blocks(x)                              # [1 + N, B, D]

        # separate CLS token and patch tokens
        cls_token, patch_tokens = x[0:1], x[1:]             # [1, B, D], [N, B, D]

        # predict boundaries
        avg_boundaries_per_batch = 0
        boundary_ratio = 0
        if self.flop_measure:
            print("=" * 80)
            print("[INFO] FLOP measurement mode: simulating fake boundaries to satisfy the compression rate.")
            print("=" * 80)
            L = patch_tokens.shape[0]
            interval = max(1, int(1 / self.compression_rate))
            hard_boundaries = torch.zeros(B, L, device=x.device)
            hard_boundaries[:, ::interval] = 1
            soft_boundaries = hard_boundaries.clone()
        else:
            soft_boundaries, hard_boundaries = self.boundary_predictor(patch_tokens)

            # [B] → count per sample
            num_boundaries = hard_boundaries.sum(dim=1)
            avg_boundaries_per_batch = num_boundaries.float().mean()
            seq_len = hard_boundaries.shape[1]
            boundary_ratio = avg_boundaries_per_batch / seq_len

        # downsample patch tokens
        patch_tokens = downsample(hard_boundaries, patch_tokens, self.null_group)
        patch_tokens = patch_tokens[1:]  # remove null group at index 0 → [S, B, D]

        # reattach CLS token
        x = torch.cat([cls_token, patch_tokens], dim=0)     # [1 + S, B, D]

        x = self.norm(x)                                     # [1 + S, B, D]
        x = self.shorten_blocks(x)                           # [1 + S, B, D]

        x = x.transpose(0, 1)                               # [B, 1 + S, D]
        x = self.norm(x)                                    # [B, 1 + S, D]
        x = x.mean(dim=1)                                   # [B, D]
        
        logits = self.head(x)                               # [B, num_classes]

        if return_loss and not self.flop_measure:
            if self.lower_bound:
                # use soft boundaries for adaptive boundary loss with lower bound
                boundary_loss = self.boundary_predictor.calc_loss_lower_bound(
                    preds=hard_boundaries,
                    #preds=soft_boundaries,
                    lambda_val=self.lambda_val
                )
            else:
                # use hard boundaries for fixed boundary loss
                boundary_loss = self.boundary_predictor.calc_loss(
                    preds=hard_boundaries, 
                    gt=None
                )
            return logits, boundary_loss, avg_boundaries_per_batch, boundary_ratio
        else:
            return logits
        
    def encode(self, x: torch.Tensor, return_loss=False): # forward, but no pooling, no head
        # input shape: [B, 3, H, W]
        B = x.size(0)

        x = self.patch_embed(x)                             # [B, N, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)       # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)               # [B, 1 + N, D]
        x = x + self.pos_embed                              # [B, 1 + N, D]
        x = self.pos_drop(x)                                # [B, 1 + N, D]
        x = x.transpose(0, 1)                               # [1 + N, B, D]

        x = self.pre_blocks(x)                              # [1 + N, B, D]

        # separate CLS token and patch tokens
        cls_token, patch_tokens = x[0:1], x[1:]             # [1, B, D], [N, B, D]

        # predict boundaries
        avg_boundaries_per_batch = 0
        boundary_ratio = 0
        if self.flop_measure:
            print("=" * 80)
            print("[INFO] FLOP measurement mode: simulating fake boundaries to satisfy the compression rate.")
            print("=" * 80)
            L = patch_tokens.shape[0]
            interval = max(1, int(1 / self.compression_rate))
            hard_boundaries = torch.zeros(B, L, device=x.device)
            hard_boundaries[:, ::interval] = 1
            soft_boundaries = hard_boundaries.clone()
        else:
            soft_boundaries, hard_boundaries = self.boundary_predictor(patch_tokens)

            # [B] → count per sample
            num_boundaries = hard_boundaries.sum(dim=1)
            avg_boundaries_per_batch = num_boundaries.float().mean()
            seq_len = hard_boundaries.shape[1]
            boundary_ratio = avg_boundaries_per_batch / seq_len

        # downsample patch tokens
        patch_tokens = downsample(hard_boundaries, patch_tokens, self.null_group)
        patch_tokens = patch_tokens[1:]  # remove null group at index 0 → [S, B, D]

        # reattach CLS token
        x = torch.cat([cls_token, patch_tokens], dim=0)     # [1 + S, B, D]

        x = self.norm(x)                                     # [1 + S, B, D]
        x = self.shorten_blocks(x)                           # [1 + S, B, D]

        x = x.transpose(0, 1)                               # [B, 1 + S, D]
        x = self.norm(x)                                    # [B, 1 + S, D]

        if return_loss and not self.flop_measure:
            if self.lower_bound:
                # use soft boundaries for adaptive boundary loss with lower bound
                boundary_loss = self.boundary_predictor.calc_loss_lower_bound(
                    preds=hard_boundaries,
                    #preds=soft_boundaries,
                    lambda_val=self.lambda_val
                )
            else:
                # use hard boundaries for fixed boundary loss
                boundary_loss = self.boundary_predictor.calc_loss(
                    preds=hard_boundaries, 
                    gt=None
                )
            return x, boundary_loss, avg_boundaries_per_batch, boundary_ratio
        else:
            return x
