from typing import Optional, Union
import torch
import transformers

from transformers.cache_utils import Cache
from transformers.utils import is_torchdynamo_compiling
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModelOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs

def replace_qwen3_with_mixed_modality_forward_inference_only():
    transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModel.forward = (
        qwen3_vl_mixed_modality_forward_inference_only
    )

def qwen3_vl_mixed_modality_forward_inference_only(
    self: transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModel,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen3VLModelOutputWithPast]:
    """
    Inference-only forward for Qwen3-VL mixed modality support.

    Key difference from training patch:
    - If pixel_values / pixel_values_videos are None (typical decode stage),
      do NOT run any dummy visual forward.
    - Instead, create empty visual metadata tensors so cached decoding stays fast.
    """

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    video_mask = None
    deepstack_image_embeds = None
    deepstack_video_embeds = None

    """
    Prefill stage: real image/video features only if actually provided
    """
    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = self.get_image_features(
            pixel_values, image_grid_thw
        )
        image_embeds = torch.cat(image_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_embeds, deepstack_video_embeds = self.get_video_features(
            pixel_values_videos, video_grid_thw
        )
        video_embeds = torch.cat(video_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    # build visual_pos_masks + deepstack_visual_embeds
    visual_pos_masks = None
    deepstack_visual_embeds = None

    if image_mask is not None and video_mask is not None:
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask

        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]

        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(
                (visual_pos_masks.sum(), img_embed.shape[-1])
            )
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)

    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds

    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    """
        Decode stage (or no visual input)
    """
    if visual_pos_masks is None:
        B, S, H = inputs_embeds.shape
        visual_pos_masks = torch.zeros(
            (B, S), dtype=torch.bool, device=inputs_embeds.device
        )

        # Create empty deepstack tensors directly.
        # Assumes deepstack features live in the LM hidden dimension H.
        num_deepstack = len(self.visual.deepstack_visual_indexes)
        deepstack_visual_embeds = [
            inputs_embeds.new_empty((0, H)) for _ in range(num_deepstack)
        ]
        
    if position_ids is None:
        attention_mask_tensor = (
            attention_mask
            if not isinstance(attention_mask, dict)
            else attention_mask["full_attention"]
        )

        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(
                attention_mask_tensor[:, 0], dim1=1, dim2=2
            )
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = (
                    attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                )
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = (not is_torchdynamo_compiling()) and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )

        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )

            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)

            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)

            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )

    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )
