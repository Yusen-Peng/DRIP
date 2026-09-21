# compression.py
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .BP import BoundaryPredictor, downsample
from .BP_alternative import new_downsample

def prunesid_pca_group(features, min_components=32):
    """
    Copied/adapted from PruneSID's Qwen2-VL implementation.

    Args:
        features: [L, D], post-Qwen spatial-merger visual features.

    Returns:
        projector_lengths: [L, num_components]
        belong_components: [L]
    """
    standard_features = torch.sigmoid(
        features.to(torch.float32)
    ).permute(1, 0)

    # pca_lowrank requires q <= min(D, L)
    q = min(
        min_components,
        standard_features.shape[0],
        standard_features.shape[1],
    )

    _, _, V = torch.pca_lowrank(
        standard_features,
        q=q,
    )

    V = torch.abs(V)
    belong_components = torch.argmax(V, dim=1)

    return V, belong_components


def prunesid_nms(similarity_matrix, scores, threshold):
    """
    Copied from PruneSID's Qwen2-VL implementation.

    Args:
        similarity_matrix: numpy array [N, N]
        scores: numpy array [N]
        threshold: scalar
    """
    keep = []

    while scores.sum() > 0:
        max_idx = scores.argmax(axis=0)

        scores[max_idx] = 0
        keep.append(max_idx)

        condition = similarity_matrix[max_idx] > threshold
        scores[condition] = 0

    return keep


class TokenCompressor(nn.Module):
    def __init__(
        self,
        hidden_size, # mega-token dim, e.g. 4096
        bp_hidden_size=None, # native patch dim, e.g. 1024
        bp_intermediate_size=None, # e.g. 4096
        merge_strategy="DRIP",
        compression_rate=0.25,
        temperature=0.1,
        drip_path=None,
    ):
        super().__init__()

        self.merge_strategy = merge_strategy
        self.compression_rate = compression_rate
        self.temperature = temperature
        self.null_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        if merge_strategy == "DRIP":
            self.boundary_predictor = BoundaryPredictor(
                d_model=bp_hidden_size,
                d_inner=bp_intermediate_size,
                activation_function="gelu",
                temp=temperature,
                prior=compression_rate,
                bp_type="gumbel",
                threshold=0.5,
                smart_init=False,
            )
            if drip_path is not None:
                self.load_drip_weights(drip_path)

        elif merge_strategy == "Fixed":
            self.boundary_predictor = None
        elif merge_strategy == "PruneSID":
            self.boundary_predictor = None
        else:
            raise ValueError(f"Unknown strategy: {merge_strategy}")

        self.last_soft_boundaries = None # keep track of the patch-level soft boundaries for debugging and analysis
        self.last_pooled_scores = None # keep track of the pooled scores for debugging and analysis
        
    def load_drip_weights(self, drip_path):
        """
            Load DRIP weights from either:

            1. Qwen-style checkpoints
            ...compressor.boundary_predictor.0.weight
            ...compressor.boundary_predictor.0.bias
            ...compressor.null_token

            2. pretrained SigLIP2-LLaVA checkpoints
            ...vision_tower.boundary_predictor.0.weight
            ...vision_tower.boundary_predictor.0.bias
            ...vision_tower.null_token
        """
        print(f"🌊🌊🌊 [INFO] Loading DRIP weights from {drip_path}")
        sd = torch.load(drip_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        bp_sd = {}
        null_tensor = None

        for k, v in sd.items():
            # 1. Legacy SigLIP2-LLaVA format
            siglip_bp_anchor = "vision_tower.boundary_predictor."

            if siglip_bp_anchor in k:
                new_k = k.split(siglip_bp_anchor, 1)[1]
                bp_sd[new_k] = v
                continue

            if k.endswith("vision_tower.null_token"):
                null_tensor = v
                continue

            # 2. Qwen compressor format
            qwen_bp_anchor = "compressor.boundary_predictor."

            if qwen_bp_anchor in k:
                new_k = k.split(qwen_bp_anchor, 1)[1]
                bp_sd[new_k] = v
                continue

            if k.endswith("compressor.null_token"):
                null_tensor = v
                continue

            # 3. Already-local compressor/BP checkpoint
            if k.startswith("boundary_predictor."):
                new_k = k.split("boundary_predictor.", 1)[1]
                bp_sd[new_k] = v
                continue
            if k == "null_token":
                null_tensor = v
                continue

        if len(bp_sd) == 0:
            raise RuntimeError(
                f"No boundary_predictor weights found in {drip_path}.\n"
                f"First checkpoint keys:\n"
                + "\n".join(f"  {k}" for k in list(sd.keys())[:20])
            )
        print("🌊🌊🌊 [INFO] Extracted boundary predictor keys:")
        for k, v in bp_sd.items():
            print(f"    {k}: {tuple(v.shape)}")
        missing, unexpected = self.boundary_predictor.load_state_dict(bp_sd, strict=True)
        print("✅ [INFO] Loaded boundary_predictor")
        if null_tensor is not None:
            if self.null_token.shape == null_tensor.shape:
                with torch.no_grad():
                    self.null_token.copy_(
                        null_tensor.to(
                            device=self.null_token.device,
                            dtype=self.null_token.dtype,
                        )
                    )
                print(f"✅ [INFO] Loaded null_token: {tuple(null_tensor.shape)}")
            else:
                print(
                    "⚠️ [INFO] Skipping null_token due to shape mismatch: "
                    f"checkpoint={tuple(null_tensor.shape)}, "
                    f"current={tuple(self.null_token.shape)}"
                )
        else:
            print("⚠️ [INFO] null_token not found in checkpoint")
        if missing:
            print(f"⚠️ [INFO] Missing BP keys: {missing}")
        if unexpected:
            print(f"⚠️ [INFO] Unexpected BP keys: {unexpected}")
        return missing, unexpected

    @torch.no_grad()
    def get_prunesid_boundaries(
        self,
        mega_features: torch.Tensor,
    ):
        """
        PruneSID adapted from the authors' Qwen2-VL implementation.

        Args:
            mega_features:
                Post-Qwen spatial-merger visual features.
                Shape: [B, L, D]

        Returns:
            pooled_boundaries:
                Binary keep mask over Qwen mega-tokens.
                Shape: [B, L]
        """
        B, L, D = mega_features.shape
        if B != 1:
            raise NotImplementedError(
                "PruneSID currently supports batch_size=1."
            )

        # Match the exact token budget used by DRIP / Fixed.
        need_token_num = max(
            1,
            int(L * self.compression_rate),
        )
        if need_token_num >= L:
            return mega_features.new_ones(B, L)
        hidden_states = mega_features[0]  # [L, D]
        num_components = max(
            int(need_token_num / 4),
            4,
        )
        # Necessary for small native-resolution images.
        num_components = min(num_components, hidden_states.shape[0], hidden_states.shape[1])
        projector_lengths, belong_components = prunesid_pca_group(hidden_states, min_components=num_components)
        projector_scores = projector_lengths.clone()

        # [L, num_components]
        projector_mask = belong_components.unsqueeze(1).repeat(1, projector_lengths.shape[1])
        index_map = torch.arange(projector_lengths.shape[1], device=projector_lengths.device).unsqueeze(0).repeat(projector_lengths.shape[0],1)
        weights_mask = torch.where(projector_mask != index_map)

        projector_lengths[weights_mask] = 0
        projector_scores[weights_mask] = 0

        normalized_states = F.normalize(hidden_states, p=2, dim=-1)

        group_similarity = torch.bmm(
            normalized_states.unsqueeze(0),
            normalized_states.T.unsqueeze(0),
        )[0]

        sim_mean = group_similarity.triu(
            diagonal=0
        ).mean()

        group_similarity_np = group_similarity.to(torch.float32).cpu().numpy()

        group_idxs = []

        ratio = max(need_token_num / (L / 18), 1)

        given_scores = torch.arange(
            projector_lengths.shape[0],
            0,
            -1,
            device=projector_lengths.device,
            dtype=projector_lengths.dtype,
        )

        for g in range(projector_lengths.shape[1]):

            group_indices = torch.where(
                belong_components == g
            )[0].cpu().numpy()

            if group_indices.shape[0] == 0:
                group_idxs.append(
                    np.array([], dtype=np.int64)
                )
                continue

            g_similarity = group_similarity_np[
                group_indices, :
            ][:, group_indices]

            g_scores = (
                projector_lengths[:, g][group_indices]
                .cpu()
                .numpy()
            )

            keep_indices = prunesid_nms(
                g_similarity,
                g_scores,
                float(ratio * sim_mean),
            )

            keep_indices = group_indices[keep_indices]

            projector_scores[
                keep_indices,
                g,
            ] = given_scores[:keep_indices.shape[0]]

            group_idxs.append(keep_indices)

        keep_nms_counts = torch.tensor(
            [
                group_idxs[i].shape[0]
                for i in range(len(group_idxs))
            ],
            device=projector_lengths.device,
        )

        group_counts = (
            projector_mask == index_map
        ).sum(dim=0)

        group_lower_bound = torch.ones(
            group_counts.shape[0],
            device=group_counts.device,
        )

        group_lower_bound = torch.min(
            torch.cat(
                [
                    group_lower_bound.unsqueeze(0),
                    group_counts.unsqueeze(0),
                ],
                dim=0,
            ),
            dim=0,
        )[0]

        group_upper_bound = torch.ones(
            group_counts.shape[0],
            device=group_counts.device,
        ) * 5 * math.ceil(need_token_num / 64)

        group_upper_bound = torch.min(
            torch.cat(
                [
                    group_upper_bound.unsqueeze(0),
                    group_counts.unsqueeze(0),
                ],
                dim=0,
            ),
            dim=0,
        )[0]

        group_upper_bound = torch.min(
            torch.cat(
                [
                    group_upper_bound.unsqueeze(0),
                    keep_nms_counts.unsqueeze(0),
                ],
                dim=0,
            ),
            dim=0,
        )[0]

        while group_upper_bound.sum() < need_token_num:
            group_upper_bound = group_upper_bound + 1

            group_upper_bound = torch.min(
                torch.cat(
                    [
                        group_upper_bound.unsqueeze(0),
                        group_counts.unsqueeze(0),
                    ],
                    dim=0,
                ),
                dim=0,
            )[0]

        other_token_nums = max(
            0,
            need_token_num - group_lower_bound.sum(),
        )

        norm_group_counts = (
            keep_nms_counts / keep_nms_counts.sum()
        )

        cumulative_sum = torch.cumsum(
            norm_group_counts,
            dim=0,
        )

        other_token_d = (
            cumulative_sum * other_token_nums
        ).round().int()

        other_token_d = other_token_d - torch.cat(
            [
                torch.zeros(
                    1,
                    device=other_token_d.device,
                ),
                other_token_d[:-1],
            ]
        )

        group_token_d = (
            other_token_d + group_lower_bound
        )

        group_token_d = torch.min(
            torch.cat(
                [
                    group_token_d.unsqueeze(0),
                    group_upper_bound.unsqueeze(0),
                ],
                dim=0,
            ),
            dim=0,
        )[0]

        group_mean_sort_index = torch.argsort(
            keep_nms_counts,
            descending=True,
            dim=0,
        )

        filling_group = 0

        target_total = (
            other_token_nums
            + group_lower_bound.sum()
        )

        while group_token_d.sum() < target_total:

            group_idx = group_mean_sort_index[
                filling_group
            ]

            filling_num = min(
                group_upper_bound[group_idx]
                - group_token_d[group_idx],
                target_total
                - group_token_d.sum(),
            )

            group_token_d[group_idx] += filling_num
            filling_group += 1

        projector_sort_index = torch.argsort(
            projector_scores,
            descending=True,
            dim=0,
        )

        important_indices = []

        for g in range(len(group_token_d)):
            num_from_group = int(
                group_token_d[g].item()
            )

            important_indices.append(
                projector_sort_index[:, g][
                    :num_from_group
                ]
            )
        important_indices = torch.cat(
            important_indices,
            dim=0,
        )[:need_token_num]
        important_indices = important_indices.sort()[0]
        pooled_boundaries = mega_features.new_zeros(B, L)
        pooled_boundaries[0, important_indices] = 1.0
        return pooled_boundaries

    def get_fixed_pooled_boundaries(self, patch_features: torch.Tensor):
        B, L, D = patch_features.shape
        if L % 4 != 0:
            raise ValueError(f"Patch sequence length must be divisible by 4, got {L}")
        pooled_L = L // 4
        num_tokens = max(1, int(pooled_L * self.compression_rate))
        indices = torch.linspace(0, pooled_L - 1, steps=num_tokens, device=patch_features.device).round().long()
        pooled_boundaries = patch_features.new_zeros(B, pooled_L)
        pooled_boundaries[:, indices] = 1
        # Every mega-token sequence must terminate.
        pooled_boundaries[:, -1] = 1
        return pooled_boundaries

    def get_drip_pooled_boundaries(self, patch_features: torch.Tensor, pool_type: str = "max", inference: bool = False):
        B, L, D = patch_features.shape
        x_t = patch_features.transpose(0, 1)
        if inference:
            patch_soft_boundaries, _ = (self.boundary_predictor.inference(x_t))
        else:
            patch_soft_boundaries, _ = (self.boundary_predictor(x_t))

        self.last_soft_boundaries = patch_soft_boundaries.detach()
        B, L = patch_soft_boundaries.shape
        if L % 4 != 0:
            raise ValueError(f"Patch sequence length must be divisible by 4, got {L}")

        grouped = patch_soft_boundaries.view(B, L // 4, 4)

        if pool_type == "max":
            pooled_scores = grouped.amax(dim=-1)
        elif pool_type == "mean":
            pooled_scores = grouped.mean(dim=-1)
        else:
            raise ValueError(f"Unknown DRIP boundary pooling type: {pool_type}")

        self.last_pooled_scores = pooled_scores.detach()
        # Number of actual Qwen mega tokens
        pooled_L = pooled_scores.shape[1]
        # Exact mega-token budget
        num_tokens = max(1, int(pooled_L * self.compression_rate))

        # Select mega-token boundaries according to BP score
        topk_idx = pooled_scores.topk(k=num_tokens, dim=-1).indices
        pooled_boundaries = pooled_scores.new_zeros(B, pooled_L)
        pooled_boundaries.scatter_(dim=1, index=topk_idx, value=1.0)
        # Every mega-token sequence must terminate.
        pooled_boundaries[:, -1] = 1
        return pooled_boundaries

    def apply_boundaries(self, x, pooled_boundaries):
        if self.merge_strategy == "PruneSID":
            if x.shape[0] != 1:
                raise NotImplementedError("PruneSID pruning currently supports batch_size=1.")
            keep = pooled_boundaries[0].bool()
            return x[:, keep, :]
        hidden = x.transpose(0, 1)
        if self.merge_strategy == "DRIP":
            shortened = new_downsample(boundaries=pooled_boundaries, hidden=hidden, null_group=self.null_token, leading_one=False)
        else:
            shortened = downsample(boundaries=pooled_boundaries, hidden=hidden, null_group=self.null_token)
        return shortened.transpose(0, 1)

    def forward(self, x: torch.Tensor, pooled_boundaries: torch.Tensor):
        compressed = self.apply_boundaries(x, pooled_boundaries)

        if self.training and self.merge_strategy.startswith("DRIP"):
            boundary_loss = self.boundary_predictor.calc_loss(pooled_boundaries)
        else:
            boundary_loss = x.new_zeros(())

        return compressed, pooled_boundaries, boundary_loss
