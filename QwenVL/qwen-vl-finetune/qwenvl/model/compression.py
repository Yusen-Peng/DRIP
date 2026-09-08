# compression.py
import torch
import torch.nn as nn

from .BP import BoundaryPredictor, downsample
from .BP_alternative import new_downsample


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
        else:
            raise ValueError(f"Unknown strategy: {merge_strategy}")

        self.last_soft_boundaries = None
        
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
