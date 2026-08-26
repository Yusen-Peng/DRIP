import torch
from typing import Optional, Union

from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    TransformersKwargs,
    is_torchdynamo_compiling,
)

from .compression import TokenCompressor


class CompressedQwen3VLModel(Qwen3VLModel):

    def __init__(self, config):
        super().__init__(config)

        self.compressor = None

    def set_compressor(
        self,
        merge_strategy="Fixed",
        compression_rate=0.25,
        temperature=0.1,
    ):
        """
        Attach Fixed / DRIP compressor after loading pretrained Qwen.
        """

        # We'll verify this dimension on the first forward.
        hidden_size = self.config.vision_config.out_hidden_size

        self.compressor = TokenCompressor(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            merge_strategy=merge_strategy,
            compression_rate=compression_rate,
            temperature=temperature,
        )

        # Put newly-created module on same device/dtype as vision output.
        param = next(self.parameters())

        self.compressor.to(
            device=param.device,
            dtype=param.dtype,
        )

        print(
            f"🌊 Qwen3VL compressor: {merge_strategy}, "
            f"rate={compression_rate}"
        )


    def compress_image_features(
        self,
        image_embeds,
        deepstack_image_embeds,
        inference=False,
    ):
        """
        MVP:
            - batch size = 1
            - exactly one image
        """

        if self.compressor is None:
            raise RuntimeError(
                "Compressor has not been initialized. "
                "Call model.set_compressor() first."
            )

        if len(image_embeds) != 1:
            raise NotImplementedError(
                "Initial implementation supports exactly one image."
            )

        # Qwen gives this image as [L, D].
        x = image_embeds[0].unsqueeze(0)  # [1, L, D]

        compressed, boundaries, boundary_loss = (
            self.compressor(
                x,
                inference=inference,
            )
        )

        # [1, S, D] -> [S, D]
        compressed = compressed.squeeze(0)

        compressed_deepstack = []

        for deep in deepstack_image_embeds:

            # Expected: [L, D]
            if deep.ndim != 2:
                raise RuntimeError(
                    f"Unexpected DeepStack shape: {deep.shape}"
                )

            deep = deep.unsqueeze(0)

            compressed_deep = (
                self.compressor.apply_boundaries(
                    deep,
                    boundaries,
                )
            )

            compressed_deepstack.append(
                compressed_deep.squeeze(0)
            )

        return (
            compressed,
            compressed_deepstack,
            boundaries,
            boundary_loss,
        )



    # NOTE: the placeholder problem to solve here
    def build_sequence_keep_mask(
        self,
        image_mask,
        boundaries,
    ):
        """
        Convert visual-token boundaries into a keep mask over the
        complete LLM sequence.

        image_mask:
            [1, seq_len]

        boundaries:
            [1, num_image_tokens]

        returns:
            keep_mask: [1, seq_len]
        """

        if image_mask.ndim != 2:
            raise RuntimeError(
                f"Expected image_mask [B, L], got {image_mask.shape}"
            )

        if image_mask.shape[0] != 1:
            raise NotImplementedError(
                "Initial implementation supports batch_size=1."
            )

        visual_positions = torch.where(
            image_mask[0]
        )[0]

        if visual_positions.numel() != boundaries.shape[1]:
            raise RuntimeError(
                "Visual token count mismatch: "
                f"{visual_positions.numel()} placeholders vs "
                f"{boundaries.shape[1]} boundaries."
            )

        # Initially retain everything.
        keep_mask = torch.ones_like(
            image_mask,
            dtype=torch.bool,
        )

        # Remove ALL original visual slots.
        keep_mask[0, visual_positions] = False

        # A boundary represents one output token.
        boundary_indices = torch.where(
            boundaries[0] > 0.5
        )[0]

        # Reuse those original locations for compressed tokens.
        kept_visual_positions = visual_positions[
            boundary_indices
        ]

        keep_mask[0, kept_visual_positions] = True

        return keep_mask
