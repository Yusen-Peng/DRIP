# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from qwenvl.model.qwen3vl_compressed import CompressedQwen3VLForConditionalGeneration, CompressedTrainer
from transformers import AutoProcessor, Trainer



MERGE_STRATEGY = "DRIP"
COMPRESSION_RATE = 0.25
TEMPERATURE = 0.1


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


# def set_model(model_args, model):
#     visual = getattr(model, "visual", None)
#     if visual is None:
#         visual = model.model.visual

#     if model_args.tune_mm_vision:
#         for n, p in visual.named_parameters():
#             p.requires_grad = True
#     else:
#         for n, p in visual.named_parameters():
#             p.requires_grad = False

#     if model_args.tune_mm_mlp:
#         for n, p in visual.merger.named_parameters():
#             p.requires_grad = True
#     else:
#         for n, p in visual.merger.named_parameters():
#             p.requires_grad = False

#     if model_args.tune_mm_llm:
#         for n, p in model.language_model.named_parameters():
#             p.requires_grad = True
#         model.lm_head.requires_grad = True
#     else:
#         for n, p in model.language_model.named_parameters():
#             p.requires_grad = False
#         model.lm_head.requires_grad = False



def set_model(model_args, model):
    """
    Configure which Qwen-VL components are trainable.

    Supports both:
        Qwen2/2.5-style:
            model.visual
            model.language_model

        Qwen3-VL-style:
            model.model.visual
            model.model.language_model
    """
    # Vision tower
    if hasattr(model, "visual"):
        visual = model.visual
    elif hasattr(model, "model") and hasattr(model.model, "visual"):
        visual = model.model.visual
    else:
        raise AttributeError(
            f"Cannot find visual module in {model.__class__.__name__}"
        )

    # Language model
    if hasattr(model, "language_model"):
        language_model = model.language_model
    elif hasattr(model, "model") and hasattr(model.model, "language_model"):
        language_model = model.model.language_model
    else:
        raise AttributeError(
            f"Cannot find language_model module in {model.__class__.__name__}"
        )

    for _, p in visual.named_parameters():
        p.requires_grad = model_args.tune_mm_vision

    if not hasattr(visual, "merger"):
        raise AttributeError(
            f"Cannot find visual.merger in {visual.__class__.__name__}"
        )

    for _, p in visual.merger.named_parameters():
        p.requires_grad = model_args.tune_mm_mlp

    for _, p in language_model.named_parameters():
        p.requires_grad = model_args.tune_mm_llm

    if hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            p.requires_grad = model_args.tune_mm_llm
    else:
        raise AttributeError(
            f"Cannot find lm_head in {model.__class__.__name__}"
        )

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    model = CompressedQwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        dtype=torch.bfloat16,
    )

    model.model.set_compressor(
        merge_strategy=MERGE_STRATEGY, 
        compression_rate=COMPRESSION_RATE, 
        temperature=TEMPERATURE,
        drip_path=None # explicitly set to None; when training it starts uninitialized
    )

    data_args.model_type = "qwen3vl"

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        # Make sure the boundary predictor is trainable when using LoRA!
        if MERGE_STRATEGY == "DRIP":
            for name, p in model.named_parameters():
                if "compressor" in name:
                    p.requires_grad = True
        else:
            set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            # model.visual.print_trainable_parameters()
            # model.model.print_trainable_parameters()

            visual = (
                model.visual
                if hasattr(model, "visual")
                else model.model.visual
            )

            language_model = (
                model.language_model
                if hasattr(model, "language_model")
                else model.model.language_model
            )

            def print_trainable(module, name):
                total = 0
                trainable = 0
                for p in module.parameters():
                    # DeepSpeed ZeRO-3 keeps the original size here
                    numel = getattr(p, "ds_numel", p.numel())
                    total += numel
                    if p.requires_grad:
                        trainable += numel
                percentage = 100 * trainable / total if total > 0 else 0.0
                print(
                    f"🥶🥶🥶 [{name}] trainable params: "
                    f"{trainable:,} / {total:,} "
                    f"({percentage:.2f}%) 🥶🥶🥶"
                )
            
            print_trainable(visual, "Vision")
            print_trainable(language_model, "LLM")
            if hasattr(model, "lm_head"):
                print_trainable(model.lm_head, "LM Head")
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = CompressedTrainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
