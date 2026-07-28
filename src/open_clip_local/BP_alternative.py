import torch
import torch.nn.functional as F

def box(x):
    return F.relu(x) - F.relu(x - 1)

def detached_cumsum(x, dim=-1):
    """
    Computes a cumulative sum along a specified dimension where 
    the gradient of C_i only flows to x_i (past sums are detached).
    
    Args:
        x: Input tensor of arbitrary shape.
        dim: The dimension to compute the cumulative sum along.
        
    Returns:
        Tensor of the same shape as x.
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
    C = x + past_sums
    
    return C

def segments_to_matrix_diff_detached_cumsum(boundaries, upsample=False, leading_one=False):

    if leading_one:
        boundaries = boundaries.clone()
        boundaries[:, 0] = 1
        n_segments = boundaries.sum(dim=-1).max().item()
        
        if upsample:
            tmp = torch.roll(boundaries, shifts=-1, dims=1) # B x L
            tmp[:, -1] = 0
            cs = detached_cumsum(tmp, dim=-1) + 1              # (B, L)  
            cs = cs.unsqueeze(-1)                          # (B, L, 1)
            js = torch.zeros_like(tmp).unsqueeze(2) + torch.arange(n_segments, device=boundaries.device)
        else:
            cs = detached_cumsum(boundaries, dim=-1)               # (B, L)
            cs = cs.unsqueeze(-1)                          # (B, L, 1)
            js = torch.zeros_like(boundaries).unsqueeze(2) + torch.arange(n_segments, device=boundaries.device)


        mat = box(cs-js) - box(cs-js-1)
        if not upsample:
            mat = mat / (mat.sum(dim=1, keepdim=True) + 1e-9).detach()
        else:
            mat = mat / (mat.sum(dim=2, keepdim=True) + 1e-9).detach()

        return mat
    else:
        boundaries = boundaries.clone()
        boundaries[:, -1] = 1
        n_segments = boundaries.sum(dim=-1).max().item()

        if not upsample:
            # Shift boundaries right: an "end" at i means a "start" at i+1
            starts = torch.roll(boundaries, shifts=1, dims=-1)
            starts[:, 0] = 0 # The first token implicitly starts segment 0
            
            # Now cs_i = starts_i + past_sums
            # The gradient for cs_i flows perfectly to starts_i, which is b_{i-1}
            cs = detached_cumsum(starts, dim=-1) + 1 
        else:
            # Upsampling remains exactly the same, as its gradient is intact
            cs = detached_cumsum(boundaries, dim=-1) + 1

        cs = cs.unsqueeze(-1)
        js = torch.zeros_like(boundaries).unsqueeze(2) + torch.arange(n_segments, device=boundaries.device)

        mat = box(cs-js) - box(cs-js-1)
        
        if not upsample:
            mat = mat / (mat.sum(dim=1, keepdim=True) + 1e-9).detach()
        else:
            mat = mat / (mat.sum(dim=2, keepdim=True) + 1e-9).detach()

        return mat


def downsample(boundaries, hidden, null_group, leading_one=False):
    # B x L x S
    weights = segments_to_matrix_diff_detached_cumsum(boundaries, upsample=False, leading_one=leading_one)  # B x L x S

    # NOTE: the following is for autoregressive language model, where we discard the last group
    # shortened_hidden = torch.einsum('lbd,bls->sbd', hidden, weights)[:-1, ...]  # S-1 x B x D, we discard the last group because it won't be used in upsampling
    # shortened_hidden = torch.cat(
    #     [null_group.repeat(1, hidden.size(1), 1), shortened_hidden], dim=0
    # )


    # NOTE: However, we are working on a bidirectional encoder, so we keep all groups
    shortened_hidden = torch.einsum('lbd,bls->sbd', hidden, weights)  # S x B x D, we keep all groups
    return shortened_hidden

