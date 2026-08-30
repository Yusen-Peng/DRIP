# compression.py
import torch
import torch.nn as nn

from .BP import BoundaryPredictor, downsample
from .BP_alternative import new_downsample


class TokenCompressor(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        merge_strategy="DRIP",
        compression_rate=0.25,
        temperature=0.1,
        drip_path=None,
    ):
        super().__init__()

        self.merge_strategy = merge_strategy
        self.compression_rate = compression_rate
        self.temperature = temperature

        mlp_ratio = intermediate_size / hidden_size

        self.null_token = nn.Parameter(
            torch.zeros(1, 1, hidden_size)
        )

        if merge_strategy == "DRIP":
            self.boundary_predictor = BoundaryPredictor(
                d_model=hidden_size,
                d_inner=int(hidden_size * mlp_ratio),
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
            pass
        else:
            raise ValueError(f"Unknown strategy: {merge_strategy}")

    # def load_drip_weights(self, drip_path):
    #     state_dict = torch.load(drip_path, map_location="cpu")
    #     compressor_state = {}
    #     for key, value in state_dict.items():
    #         # Saved from full model.named_parameters()
    #         if "compressor." in key:
    #             key = key.split("compressor.", 1)[1]
    #         compressor_state[key] = value
    #     missing, unexpected = self.load_state_dict(compressor_state, strict=False)
    #     print(f"🌊 Loaded DRIP weights from {drip_path}")
    #     if missing:
    #         print(f"⚠️ Missing DRIP keys: {missing}")
    #     if unexpected:
    #         print(f"⚠️ Unexpected DRIP keys: {unexpected}")

    def load_drip_weights(self, drip_path):
        """
            Load DRIP weights from either:

            1. Qwen-style checkpoints
            ...compressor.boundary_predictor.0.weight
            ...compressor.boundary_predictor.0.bias
            ...compressor.null_token

            2. Legacy SigLIP2-LLaVA checkpoints
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
            if self.null_token.shape != null_tensor.shape:
                raise RuntimeError(
                    "null_token shape mismatch:\n"
                    f"    checkpoint: {tuple(null_tensor.shape)}\n"
                    f"    current:    {tuple(self.null_token.shape)}"
                )
            with torch.no_grad():
                self.null_token.copy_(null_tensor.to(device=self.null_token.device, dtype=self.null_token.dtype))
            print(f"✅ [INFO] Loaded null_token: {tuple(null_tensor.shape)}")
        else:
            print("⚠️ [INFO] null_token not found in checkpoint")
        if missing:
            print(f"⚠️ [INFO] Missing BP keys: {missing}")
        if unexpected:
            print(f"⚠️ [INFO] Unexpected BP keys: {unexpected}")
        return missing, unexpected


    def get_boundaries(self, x, inference=False):
        """
        x: [B, L, D]

        returns:
            boundaries: [B, L]
        """
        B, L, D = x.shape

        if self.merge_strategy == "Fixed":
            num_tokens = max(
                1,
                int(L * self.compression_rate),
            )

            indices = torch.linspace(
                0,
                L - 1,
                steps=num_tokens,
                device=x.device,
            ).round().long()

            boundaries = x.new_zeros(B, L)

            boundaries[:, indices] = 1

        else:
            x_t = x.transpose(0, 1)

            if inference:
                _, boundaries = (self.boundary_predictor.inference(x_t))
            else:
                _, boundaries = (self.boundary_predictor(x_t))

        # Every sequence must terminate.
        boundaries = torch.cat(
            [
                boundaries[:, :-1],
                torch.ones_like(boundaries[:, -1:]),
            ],
            dim=1,
        )

        return boundaries

    def apply_boundaries(
        self,
        x,
        boundaries,
    ):
        """
        x:          [B, L, D]
        boundaries:[B, L]
        """
        hidden = x.transpose(0, 1)

        if self.merge_strategy == "DRIP":
            shortened = new_downsample(
                boundaries=boundaries,
                hidden=hidden,
                null_group=self.null_token,
                leading_one=False,
            )
        else:
            shortened = downsample(
                boundaries=boundaries,
                hidden=hidden,
                null_group=self.null_token,
            )

        return shortened.transpose(0, 1)

    def forward(self, x: torch.Tensor, inference: bool = False):
        boundaries = self.get_boundaries(
            x,
            inference=inference,
        )

        compressed = self.apply_boundaries(
            x,
            boundaries,
        )

        if self.training and self.merge_strategy.startswith("DRIP"):
            boundary_loss = self.boundary_predictor.calc_loss(boundaries)
        else:
            boundary_loss = x.new_zeros(())
        return compressed, boundaries, boundary_loss

