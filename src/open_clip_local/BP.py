import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
from dataclasses import dataclass

def downsample(boundaries: torch.Tensor, hidden: torch.Tensor, null_group: torch.Tensor):
    B, L = boundaries.shape
    _, _, D = hidden.shape

    boundaries = boundaries.to(dtype=torch.long).clone()  # [B, L]

    # Number of segments per example and across the batch
    seg_counts = boundaries.sum(dim=1)                    # [B]
    S = int(seg_counts.max().item())

    # If no segments at all in the batch, return a single null segment
    if S == 0:
        # shape [1, B, D]
        return null_group.expand(1, B, D).to(hidden.dtype).to(hidden.device)

    # Build [B, L, S] template of segment indices 0..S-1
    seg_ids = torch.arange(S, device=boundaries.device).view(1, 1, S)        # [1,1,S]
    seg_ids = seg_ids.expand(B, L, S)                                        # [B,L,S]

    # Segment index for each token position: 0,0,0,1,1,2,... (per-example)
    # cumulative_num_boundaries counts boundaries up to and including pos i
    cumulative = boundaries.cumsum(dim=1)                                    # [B,L]
    real_segment_index = cumulative - boundaries                             # [B,L]

    # One-hot membership mask: token at (b, l) belongs to segment k iff k == real_segment_index[b,l]
    membership = (real_segment_index.unsqueeze(-1) == seg_ids).to(hidden.dtype)  # [B,L,S]

    # Normalize over L so each segment’s weights sum to 1
    denom = membership.sum(dim=1, keepdim=True).clamp_min(1e-9)              # [B,1,S]
    weights = membership / denom                                             # [B,L,S]

    # Weighted average over tokens -> [S, B, D]
    shortened_hidden = torch.einsum('lbd,bls->sbd', hidden, weights)
    return shortened_hidden

class BoundaryPredictor(nn.Module):
    def __init__(self, d_model, d_inner, activation_function,
                 temp, prior, bp_type, threshold=0.5, smart_init=False,
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

        if smart_init:
            first = self.boundary_predictor[0]
            last = self.boundary_predictor[2]
            # standard initialization for the first weight
            nn.init.xavier_uniform_(first.weight)
            # zero bias - let BP learn
            nn.init.zeros_(first.bias)
            # zero weight + positive bias
            # start with all tokens as boundaries, let BP learn to remove them
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, 5.0)
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, hidden, verbose: bool = False):
        # Hidden is of shape [seq_len x bs x d_model]
        # Boundaries we return are [bs x seq_len]

        boundary_logits = self.boundary_predictor(hidden).squeeze(-1).transpose(0, 1)
        boundary_probs = torch.sigmoid(boundary_logits)

        bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
            temperature=self.temp,
            probs=boundary_probs,
        )

        soft_boundaries = bernoulli.rsample()

        hard_boundaries = (soft_boundaries > self.threshold).float()
        hard_boundaries = (
            hard_boundaries - soft_boundaries.detach() + soft_boundaries
        )

        if verbose:
            print("================================")
            np.set_printoptions(suppress=True, precision=4)
            print(f"raw logits:")
            print(boundary_logits.cpu().numpy())
            print(f"probabilities after sigmoid:")
            print(boundary_probs.cpu().numpy())
            print(f"probabilities after sampling:")
            print(soft_boundaries.cpu().numpy())
            print(f"hard boundaries after thresholding:")
            print(hard_boundaries.cpu().numpy())
            print("================================")

        return soft_boundaries, hard_boundaries

    def calc_loss(self, preds):
        # B x T
        total_count = preds.size(-1)
        target_count = preds.sum(dim=-1)
        binomial = torch.distributions.binomial.Binomial(
            total_count=total_count,
            probs=torch.Tensor([self.prior]).to(preds.device)
        )
        loss_boundaries = -binomial.log_prob(target_count).mean() / total_count
        return loss_boundaries



    def inference(self, hidden, verbose: bool = False):
        # hidden: [T, B, D]
        boundary_logits = self.boundary_predictor(hidden).squeeze(-1).transpose(0, 1)  # [B, T]
        boundary_probs = torch.sigmoid(boundary_logits)  # [B, T]
        
        soft_boundaries = boundary_probs

        _, T = soft_boundaries.shape
        k = int(round(T * self.compression_rate))
        k = max(1, min(k, T))  # safety clamp

        # top-k routing: select exactly k boundaries per sample
        _, topk_idx = torch.topk(soft_boundaries, k=k, dim=-1)

        hard_boundaries = torch.zeros_like(soft_boundaries)
        hard_boundaries.scatter_(dim=-1, index=topk_idx, value=1.0)

        if verbose:
            print("================================")
            np.set_printoptions(suppress=True, precision=4)
            print(f"compression_rate: {self.compression_rate}")
            print(f"T: {T}, target k: {k}")
            print(f"raw logits:")
            print(boundary_logits.cpu().numpy())
            print(f"probabilities after sigmoid:")
            print(boundary_probs.cpu().numpy())
            print(f"hard boundaries after top-k:")
            print(hard_boundaries.cpu().numpy())
            print(f"num boundaries per sample:")
            print(hard_boundaries.sum(dim=-1).cpu().numpy())
            print("================================")

        return soft_boundaries, hard_boundaries

"""
    The following code is carefully adapted from H-Net (ICLR 2026):
    https://github.com/goombalab/hnet/blob/main/hnet/modules/dc.py
"""


class H_Net(BoundaryPredictor):
    def __init__(
        self, d_model, d_inner, activation_function,
        temp, prior, bp_type, threshold=0.5, smart_init=False,
        image_size=None, patch_size=None, embed_dim=None,
        device=None, dtype=None
    ):
        super().__init__(
            d_model=d_model,
            d_inner=d_inner,
            activation_function=activation_function,
            temp=temp,
            prior=prior,
            bp_type=bp_type,
            threshold=threshold,
            smart_init=smart_init,
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        eye_tensor = torch.eye(d_model, **factory_kwargs)
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(eye_tensor)
            self.k_proj_layer.weight.copy_(eye_tensor)
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def _compute_keep_prob(self, hidden_states):
        # hidden_states: [L, B, D]        
        # [B, L, D]
        hidden_states = hidden_states.transpose(0, 1)  
        # [B, L-1, D]
        q = F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1, eps=1e-6)
        # [B, L-1, D]
        k = F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1, eps=1e-6)
        # [B, L-1]
        cos_sim = torch.einsum("bld,bld->bl", q, k).clamp(-1.0, 1.0)
        # [B, L-1]
        transition_prob = ((1.0 - cos_sim) / 2.0).clamp(0.0, 1.0)
        # [B, 1]
        last_prob = torch.ones(transition_prob.size(0), 1, device=transition_prob.device, dtype=transition_prob.dtype)
        # [B, L-1] + [B, 1] -> [B, L]
        keep_prob = torch.cat([transition_prob, last_prob], dim=1)
        return keep_prob

    def forward(self, hidden_states, verbose=False):
        keep_prob = self._compute_keep_prob(hidden_states)
        bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
            temperature=self.temp,
            probs=keep_prob,
        )
        soft_boundaries = bernoulli.rsample()                 # [B, L]
        hard_boundaries = (soft_boundaries > self.threshold).float()
        hard_boundaries = hard_boundaries - soft_boundaries.detach() + soft_boundaries
        return soft_boundaries, hard_boundaries

    def inference(self, hidden_states, verbose=False):
        soft_boundaries = self._compute_keep_prob(hidden_states)  # [B, L]

        _, T = soft_boundaries.shape
        k = int(round(T * self.compression_rate))
        k = max(1, min(k, T))  # safety clamp

        # top-k routing: select exactly k boundaries per sample
        _, topk_idx = torch.topk(soft_boundaries, k=k, dim=-1)

        hard_boundaries = torch.zeros_like(soft_boundaries)
        hard_boundaries.scatter_(dim=-1, index=topk_idx, value=1.0)

        if verbose:
            print("================================")
            np.set_printoptions(suppress=True, precision=4)
            print(f"compression_rate: {self.compression_rate}")
            print(f"T: {T}, target k: {k}")
            print(f"raw logits:")
            print(hard_boundaries.cpu().numpy())
            print(f"num boundaries per sample:")
            print(hard_boundaries.sum(dim=-1).cpu().numpy())
            print("================================")

        return soft_boundaries, hard_boundaries


##################################################################################
##################################################################################


class RoutingModule(nn.Module):

    def __init__(self, prior, d_model, device=None, dtype=None):
        self.prior = prior
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        
        # # NOTE: apply top-k routing
        # with torch.no_grad():
        #     self.q_proj_layer.weight.copy_(torch.eye(d_model))
        #     self.k_proj_layer.weight.copy_(torch.eye(d_model))
        
        # # NOTE: apply thresholding routing with zero initialization
        # with torch.no_grad():
        #     self.q_proj_layer.weight.zero_()
        #     self.k_proj_layer.weight.zero_()
        with torch.no_grad():
            nn.init.normal_(self.q_proj_layer.weight, mean=0.0, std=1e-3)
            nn.init.normal_(self.k_proj_layer.weight, mean=0.0, std=1e-3)


        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def forward(self, hidden_states: torch.Tensor):  # [L, B, D]
        hidden_states = hidden_states.transpose(0, 1)  # [B, L, D]
        q = F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1, eps=1e-6)
        k = F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1, eps=1e-6)
        cos_sim = torch.einsum("b l d, b l d -> b l", q, k)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        keep_prob = torch.clamp((1.0 - cos_sim) / 2.0, min=0.0, max=1.0)   # [B, L-1]
        # Force first token to always be kept
        keep_prob = F.pad(keep_prob, (1, 0), "constant", 1.0)               # [B, L]
        # 2-class probability tensor, same format as before
        boundary_prob = torch.stack((1.0 - keep_prob, keep_prob), dim=-1)   # [B, L, 2]

        # # NOTE: apply top-k routing
        # B, L = keep_prob.shape
        # k = max(1, int(round(L * self.prior)))

        # # reserve one slot for it when token 0 is forced kept
        # num_extra = max(0, k - 1)

        # boundary_mask = torch.zeros_like(keep_prob)                          # [B, L]
        # boundary_mask[:, 0] = 1.0
        # if num_extra > 0 and L > 1:
        #     scores = keep_prob[:, 1:] # rank tokens 1..L-1
        #     topk_idx = torch.topk(scores, k=min(num_extra, L - 1), dim=1).indices
        #     boundary_mask[:, 1:].scatter_(1, topk_idx, 1.0)
        # return boundary_prob.to(hidden_states.dtype), boundary_mask.to(hidden_states.dtype)
    

        # NOTE: apply thresholding routing (>=0.5) with zero initialization
        boundary_mask = (keep_prob >= 0.5).float()                                # [B, L]
        return boundary_prob.to(hidden_states.dtype), boundary_mask.to(hidden_states.dtype)

    def calc_loss(self, boundary_prob: torch.Tensor, boundary_mask: torch.Tensor) -> torch.Tensor:
        """
        boundary_prob: [B, L, 2]
        boundary_mask: [B, L]
        """
        soft_keep = boundary_prob[..., 1].float()   # [B, L]
        hard_keep = boundary_mask.float()           # [B, L]

        F_ratio = hard_keep.mean(dim=-1)            # [B]
        G_ratio = soft_keep.mean(dim=-1)            # [B]

        N = int(1.0 / self.prior)

        loss_ratio = (
            ((N - 1.0) * F_ratio * G_ratio) +
            ((1.0 - F_ratio) * (1.0 - G_ratio))
        ) * (N / (N - 1.0))

        return loss_ratio.mean()



def downsample_with_indices(boundaries: torch.Tensor, hidden: torch.Tensor, null_group: torch.Tensor):
    B, L = boundaries.shape
    _, _, D = hidden.shape

    boundaries = boundaries.to(dtype=torch.long).clone()  # [B, L]

    # Number of segments per example and across the batch
    seg_counts = boundaries.sum(dim=1)                    # [B]
    S = int(seg_counts.max().item())

    # If no segments at all in the batch, return a single null segment
    # FIXME: there is a bug here
    if S == 0:
        # shape [1, B, D]
        return null_group.expand(1, B, D).to(hidden.dtype).to(hidden.device)

    # Build [B, L, S] template of segment indices 0..S-1
    seg_ids = torch.arange(S, device=boundaries.device).view(1, 1, S)        # [1,1,S]
    seg_ids = seg_ids.expand(B, L, S)                                        # [B,L,S]

    # Segment index for each token position: 0,0,0,1,1,2,... (per-example)
    # cumulative_num_boundaries counts boundaries up to and including pos i
    cumulative = boundaries.cumsum(dim=1)                                    # [B,L]
    real_segment_index = cumulative - boundaries                             # [B,L]

    # One-hot membership mask: token at (b, l) belongs to segment k iff k == real_segment_index[b,l]
    membership = (real_segment_index.unsqueeze(-1) == seg_ids).to(hidden.dtype)  # [B,L,S]

    # Normalize over L so each segment’s weights sum to 1
    denom = membership.sum(dim=1, keepdim=True).clamp_min(1e-9)              # [B,1,S]
    weights = membership / denom                                             # [B,L,S]

    # Weighted average over tokens -> [S, B, D]
    shortened_hidden = torch.einsum('lbd,bls->sbd', hidden, weights)

    rep_idx = membership.argmax(dim=1).to(torch.long) # NOTE: the key line
    return shortened_hidden, rep_idx
