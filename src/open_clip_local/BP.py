from numpy import eye
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

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


"""
    The following code is carefully adapted from H-Net (ICLR 2026):
    https://github.com/goombalab/hnet/blob/main/hnet/modules/dc.py
"""

class RoutingModule(nn.Module):

    def __init__(self, prior, d_model, device=None, dtype=None):
        super().__init__()
        self.prior = prior
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}


        # corresponds to qt = Wq * x^t , kt = Wk * x^t in the paper
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        with torch.no_grad():
            # initial routing = raw cosine similarity between hidden states
            eye_mat = torch.eye(d_model, device=self.q_proj_layer.weight.device, dtype=self.q_proj_layer.weight.dtype)
            self.q_proj_layer.weight.copy_(eye_mat)
            self.k_proj_layer.weight.copy_(eye_mat)


        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def forward(self, hidden_states: torch.Tensor): # [L, B, D]
        # hidden_states: [L, B, D]
        hidden_states = hidden_states.transpose(0, 1)  # -> [B, L, D]

        # corresponds to the cosine similarity in the paper
        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
        )
        # corresponds to the boundary probabilities in the paper
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        PAD_PROB = 1.0 # Force boundary probability of the first element to 1.0
        boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)

        # corresponds to thresholding bt = 1{pt ≥ 0.5} in the paper
        boundary_mask = boundary_prob >= 0.5

        return boundary_prob, boundary_mask

    def calc_loss(self, preds: torch.Tensor):
        """We need this to compute the axillary loss for the boundary predictor"""
        preds = preds.float()
        total_count = preds.size(-1)              # L
        target_count = preds.sum(dim=-1)          # [B]
        binomial = torch.distributions.Binomial(
            total_count=total_count,
            probs=torch.tensor(self.prior, device=preds.device, dtype=preds.dtype)
        )
        loss_boundaries = -binomial.log_prob(target_count).mean() / total_count
        return loss_boundaries
