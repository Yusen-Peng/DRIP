# compression.py

import torch
import torch.nn as nn
import os, sys


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../../../"))
sys.path.insert(0, PROJECT_ROOT)

from src.open_clip_local.BP import BoundaryPredictor, downsample, H_Net
from src.open_clip_local.BP_alternative import new_downsample


class TokenCompressor(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        merge_strategy="DRIP",
        compression_rate=0.25,
        temperature=0.1,
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

        elif merge_strategy == "DRIP-H":
            self.boundary_predictor = H_Net(
                d_model=hidden_size,
                d_inner=int(hidden_size * mlp_ratio),
                activation_function="gelu",
                temp=temperature,
                prior=compression_rate,
                bp_type="gumbel",
                threshold=0.5,
                smart_init=False,
            )

        elif merge_strategy == "Fixed":
            pass

        else:
            raise ValueError(
                f"Unknown strategy: {merge_strategy}"
            )

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
                _, boundaries = (
                    self.boundary_predictor.inference(x_t)
                )
            else:
                _, boundaries = (
                    self.boundary_predictor(x_t)
                )

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
            boundary_loss = (
                self.boundary_predictor.calc_loss(boundaries)
            )
        else:
            boundary_loss = x.new_zeros(())

        return compressed, boundaries, boundary_loss

