import torch
from typing import Optional, Union
import os
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    TransformersKwargs,
    is_torchdynamo_compiling,
    Qwen3VLForConditionalGeneration,
    Qwen3VLCausalLMOutputWithPast
)

from transformers import Trainer
from dataclasses import dataclass

from .compression import TokenCompressor


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return



class CompressedTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        outputs = model(**inputs)

        loss = outputs.loss

        # Handle PEFT / DeepSpeed wrappers
        base_model = model
        boundary_loss = None
        # easiest: temporarily stash boundary_loss on output if available
        if hasattr(outputs, "boundary_loss"):
            boundary_loss: torch.Tensor = outputs.boundary_loss
        if boundary_loss is not None:
            self._last_boundary_loss = boundary_loss.detach().float().item()
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if hasattr(self, "_last_boundary_loss"):
            logs["boundary_loss"] = self._last_boundary_loss
        return super().log(logs, *args, **kwargs)

    def _save_checkpoint(self, model, trial):
        # First let HF / DeepSpeed / PEFT save everything normally
        super()._save_checkpoint(model, trial)

        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = (
            f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        )
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        # Save DRIP-specific parameters separately
        drip_keys_to_match = ["boundary_predictor", "null_token"]
        drip_weight_to_save = get_mm_adapter_state_maybe_zero_3(
            self.model.named_parameters(),
            drip_keys_to_match,
        )
        if self.args.local_rank in (0, -1):
            if len(drip_weight_to_save) > 0:
                torch.save(drip_weight_to_save, os.path.join(output_dir, "drip.bin"))
                print(f"🌊 Saved DRIP weights to {os.path.join(output_dir, 'drip.bin')}")

@dataclass
class CompressedQwen3VLModelOutput(Qwen3VLModelOutputWithPast):
    keep_mask: Optional[torch.Tensor] = None
    boundary_loss: Optional[torch.Tensor] = None

class CompressedQwen3VLModel(Qwen3VLModel):

    def __init__(self, config):
        super().__init__(config)
        self.compressor = None

    def set_compressor(
        self,
        merge_strategy="Fixed",
        compression_rate=0.25,
        temperature=0.1,
        drip_path=None,
    ):
        """
        attach Fixed / DRIP compressor after loading pretrained Qwen.
        """

        hidden_size = self.config.vision_config.out_hidden_size

        self.compressor = TokenCompressor(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            merge_strategy=merge_strategy,
            compression_rate=compression_rate,
            temperature=temperature,
            drip_path=drip_path
        )

        # Put newly-created module on same device/dtype as vision output.
        param = next(self.parameters())
        self.compressor.to(device=param.device, dtype=param.dtype)
        print(f"🌊 Qwen3VL compressor: {merge_strategy}, rate={compression_rate}")
        if merge_strategy == "DRIP":
            print(f"🤡🤡🤡 sampling temperature: {temperature}")


    def compress_image_features(
        self,
        image_embeds,
        deepstack_image_embeds,
        inference=False,
    ):
        """MVP: batch size = 1 with exactly one image"""

        if self.compressor is None:
            raise RuntimeError(
                "Compressor has not been initialized. "
                "Call model.set_compressor() first."
            )

        if len(image_embeds) != 1:
            raise NotImplementedError(
                "Initial implementation supports exactly one image."
            )

        """compress the features from the final layer"""
        # Qwen gives this image as [L, D]
        x = image_embeds[0].unsqueeze(0)  # [1, L, D]
        compressed, boundaries, boundary_loss = self.compressor(x, inference=inference)
        compressed = compressed.squeeze(0)


        """compress the features from the deepstack layers"""
        compressed_deepstack = []
        for deep in deepstack_image_embeds:
            # Expected: [L, D]
            if deep.ndim != 2:
                raise RuntimeError(f"Unexpected DeepStack shape: {deep.shape}")

            deep = deep.unsqueeze(0)
            compressed_deep = self.compressor.apply_boundaries(deep, boundaries)

            compressed_deepstack.append(compressed_deep.squeeze(0))

        return (
            compressed,
            compressed_deepstack,
            boundaries,
            boundary_loss
        )

    def build_sequence_keep_mask(
        self,
        image_mask,
        boundaries,
    ):
        """Convert visual-token boundaries into a keep mask over the complete LLM sequence."""

        if image_mask.ndim != 2:
            raise RuntimeError(f"Expected image_mask [B, L], got {image_mask.shape}")

        if image_mask.shape[0] != 1:
            raise NotImplementedError("Initial implementation supports batch_size=1.")

        visual_positions = torch.where(image_mask[0])[0]

        if visual_positions.numel() != boundaries.shape[1]:
            raise RuntimeError(f"🤬🤬🤬 Visual token count mismatch: {visual_positions.numel()} placeholders vs {boundaries.shape[1]} boundaries.")

        keep_mask = torch.ones_like(
            image_mask,
            dtype=torch.bool,
        )
        keep_mask[0, visual_positions] = False

        # A boundary represents one output token.
        boundary_indices = torch.where(
            boundaries[0] > 0.5
        )[0]

        # Reuse those original locations for compressed tokens.
        kept_visual_positions = visual_positions[boundary_indices]
        keep_mask[0, kept_visual_positions] = True
        return keep_mask

    def forward(
        self,
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
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, CompressedQwen3VLModelOutput]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            original_image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds,image_features=original_image_embeds
            )
            # num_placeholder_elements = int(image_mask.sum().item())
            # num_feature_elements = int(original_image_embeds.numel())
            # if num_placeholder_elements != num_feature_elements:
            #     raise RuntimeError(
            #         f"🤬 PRE-COMPRESSION scatter mismatch:\n"
            #         f"image_mask shape={image_mask.shape}\n"
            #         f"image_mask true={num_placeholder_elements}\n"
            #         f"original_image_embeds shape={original_image_embeds.shape}\n"
            #         f"source elements={num_feature_elements}\n"
            #         f"image_grid_thw={image_grid_thw}"
            #     )

            inputs_embeds = inputs_embeds.masked_scatter(image_mask, original_image_embeds)

        # no video compression yet
        if pixel_values_videos is not None:
            raise NotImplementedError("Compressed Qwen3VL currently supports images only.")

        visual_pos_masks = None
        deepstack_visual_embeds = None

        if image_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds

        if position_ids is None:
            attention_mask_tensor = (attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"])

            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0],dim1=1,dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor/ torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding 
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
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
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        """NEW: compression"""
        boundary_loss = None
        keep_mask = None

        if pixel_values is not None:
            (
                compressed_image_embeds,
                compressed_deepstack,
                boundaries,
                boundary_loss,
            ) = self.compress_image_features(
                image_embeds,
                deepstack_image_embeds,
                inference=not self.training,
            )

            keep_mask = self.build_sequence_keep_mask(image_mask, boundaries)

            # for now we only support batch size = 1;
            # so we can just index into the first row
            keep = keep_mask[0]
            input_ids = input_ids[:, keep]
            inputs_embeds = inputs_embeds[:, keep, :]
            position_ids = position_ids[:, :, keep]
            if attention_mask is not None:
                if attention_mask.ndim == 2:
                    # Standard token-level attention mask: [B, L]
                    attention_mask = attention_mask[:, keep]

                elif attention_mask.ndim == 1:
                    # Flattened / FlashAttention varlen representation (cu_seqlens).
                    # MVP supports exactly one sequence, so [0, old_L] -> [0, new_L].
                    if attention_mask.numel() != 2:
                        raise NotImplementedError(
                            f"Expected single-sequence cu_seqlens with 2 entries, "
                            f"got shape {attention_mask.shape}: {attention_mask}"
                        )

                    new_seq_len = int(keep.sum().item())

                    attention_mask = torch.tensor(
                        [0, new_seq_len],
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    )

                else:
                    raise RuntimeError(
                        f"Unexpected attention_mask shape: {attention_mask.shape}"
                    )

            visual_pos_masks = image_mask[:, keep]


            num_visual_tokens = int(visual_pos_masks.sum().item())
            num_boundary_tokens = int((boundaries[0] > 0.5).sum().item())
            num_compressed_tokens = int(compressed_image_embeds.shape[0])
            if not (
                num_visual_tokens
                == num_boundary_tokens
                == num_compressed_tokens
            ):
                raise RuntimeError(
                    f"🤬 POST-COMPRESSION mismatch:\n"
                    f"visual slots={num_visual_tokens}\n"
                    f"boundaries={num_boundary_tokens}\n"
                    f"compressed={num_compressed_tokens}\n"
                    f"boundary sum={boundaries.sum().item()}\n"
                    f"boundary shape={boundaries.shape}\n"
                    f"compressed shape={compressed_image_embeds.shape}"
                )

            compressed_image_embeds = compressed_image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            # visual_pos_masks is [B, L']
            # masked_scatter expects mask compatible with [B, L', D]
            scatter_mask = visual_pos_masks.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(scatter_mask, compressed_image_embeds)
            deepstack_visual_embeds = compressed_deepstack
            num_visual_tokens = int(visual_pos_masks.sum().item())

            assert (num_visual_tokens == compressed_image_embeds.shape[0]), (f"🤬 Visual mismatch: {num_visual_tokens} slots vs {compressed_image_embeds.shape[0]} features")
            for i, deep in enumerate(deepstack_visual_embeds):
                assert deep.shape[0] == num_visual_tokens, (
                    f"🤬 DeepStack {i}: "
                    f"{deep.shape[0]} vs "
                    f"{num_visual_tokens}"
                )
            assert (inputs_embeds.shape[1] == position_ids.shape[-1])
            # if attention_mask is not None:
            #     assert (inputs_embeds.shape[1] == attention_mask.shape[-1])

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

        return CompressedQwen3VLModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
            # NOTE: our additions
            keep_mask=keep_mask,
            boundary_loss=boundary_loss,
        )


@dataclass

class CompressedQwen3VLCausalLMOutput(Qwen3VLCausalLMOutputWithPast):
    boundary_loss: Optional[torch.Tensor] = None


class CompressedQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = CompressedQwen3VLModel(config)
        self.boundary_loss_weight = 1.0
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
            TODO: Add example
        """
        outputs: CompressedQwen3VLModelOutput = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        if labels is not None and outputs.keep_mask is not None:
            labels = labels[:, outputs.keep_mask[0]]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)
            """
                DRIP boundary regularization.
            """
            if outputs.boundary_loss is not None:
                loss = loss + self.boundary_loss_weight * outputs.boundary_loss

        return CompressedQwen3VLCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
            boundary_loss=outputs.boundary_loss,
        )
