import torch
import torch.nn.functional as F

def box(x):
    """
    a differentiable ramp from 0 to 1
    mathematically equivalent to torch.clamp(x, min=0, max=1)
    """
    return F.relu(x) - F.relu(x - 1)

def detached_cumsum(x, dim=-1):
    """
    Computes a cumulative sum along a specified dimension where 
    the gradient of C_i only flows to x_i (past sums are detached).
    """
    # 1. Shift the sequence right along the target dimension
    x_past = torch.roll(x, shifts=1, dims=dim)
    
    # 2. Dynamically construct the slice to zero out the first index
    # This creates the equivalent of [:, :, 0, :, :] for the arbitrary dim
    slices = [slice(None)] * x.dim()
    slices[dim] = 0 
    
    # Apply the slice to zero the first element along the dimension
    x_past[tuple(slices)] = 0.0
        
    # 3. Cumsum the past and sever the gradients entirely
    past_sums = torch.cumsum(x_past, dim=dim).detach()
    
    # 4. Add the current element back in
    C = x + past_sums  # still the same cumsum, but now the gradient for C_i only flows to x_i!!
    
    return C

def segments_to_matrix_diff_detached_cumsum(boundaries: torch.Tensor, leading_one: bool = False):
    if leading_one:
        boundaries = boundaries.clone()
        boundaries[:, 0] = 1
        n_segments = boundaries.sum(dim=-1).max().item()
        cs = detached_cumsum(boundaries, dim=-1)               # (B, L)
        cs = cs.unsqueeze(-1)                          # (B, L, 1)
        js = torch.zeros_like(boundaries).unsqueeze(2) + torch.arange(n_segments, device=boundaries.device)
        mat = box(cs-js) - box(cs-js-1)
        mat = mat / (mat.sum(dim=1, keepdim=True) + 1e-9).detach()
        return mat

    boundaries = boundaries.clone()
    boundaries[:, -1] = 1
    n_segments = boundaries.sum(dim=-1).max().item()

    # Shift boundaries right: an "end" at i means a "start" at i+1
    starts = torch.roll(boundaries, shifts=1, dims=-1)
    starts[:, 0] = 0 # The first token implicitly starts segment 0

    
    # Now cs_i = starts_i + past_sums
    # The gradient for cs_i flows perfectly to starts_i, which is b_{i-1}
    cs = detached_cumsum(starts, dim=-1) + 1 


    # differentiable box function to derive the same one-hot vectors
    cs = cs.unsqueeze(-1)
    js = torch.zeros_like(boundaries).unsqueeze(2) + torch.arange(n_segments, device=boundaries.device)
    mat = box(cs-js) - box(cs-js-1)
    mat = mat / (mat.sum(dim=1, keepdim=True) + 1e-9).detach()

    return mat


def new_downsample(boundaries: torch.Tensor, hidden: torch.Tensor, null_group: torch.Tensor, leading_one: bool = False):
    B, _ = boundaries.shape
    _, _, D = hidden.shape
    # Number of segments per example and across the batch
    seg_counts = boundaries.sum(dim=1)                    # [B]
    S = int(seg_counts.max().item())
    # If no segments at all in the batch, return a single null segment
    if S == 0:
        # shape [1, B, D]
        return null_group.expand(1, B, D).to(hidden.dtype).to(hidden.device)

    # B x L x S
    weights = segments_to_matrix_diff_detached_cumsum(boundaries, leading_one)  # B x L x S
    weights = weights.to(dtype=hidden.dtype)
    shortened_hidden = torch.einsum('lbd,bls->sbd', hidden, weights)  # S x B x D, we keep all groups
    return shortened_hidden
