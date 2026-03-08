import torch
import os
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from argparse import ArgumentParser
from transformers import (
    AutoProcessor,
    AutoConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

from utils.data_utils import (
    load_yaml,
    construct_prompt,
    save_json,
    process_single_sample,
    CAT_SHORT2LONG,
)
from utils.eval_utils import parse_multi_choice_response
from FT_QwenVL.src.train.monkey_patch_forward import (
    replace_qwen3_with_mixed_modality_forward,
    replace_qwen2_5_with_mixed_modality_forward,
    replace_qwen_2_with_mixed_modality_forward,
    replace_qwen3_vl_moe_with_mixed_modality_forward,
    replace_qwen3_with_mixed_modality_forward_fixed_pooling,
    replace_qwen3_with_mixed_modality_forward_drip_pooling,
)

def qwen_build_messages(sample, min_pixels=None, max_pixels=None):
    content = []

    if sample.get("image", None) is not None:
        if isinstance(sample["image"], list):
            for img in sample["image"]:
                item = {
                    "type": "image",
                    "image": img,
                }
                if min_pixels is not None:
                    item["min_pixels"] = min_pixels
                if max_pixels is not None:
                    item["max_pixels"] = max_pixels
                content.append(item)
        else:
            item = {
                "type": "image",
                "image": sample["image"],
            }
            if min_pixels is not None:
                item["min_pixels"] = min_pixels
            if max_pixels is not None:
                item["max_pixels"] = max_pixels
            content.append(item)
    content.append({"type": "text", "text": sample["final_input_prompt"]})
    messages = [{"role": "user", "content": content}]
    return messages

def qwen_prepare_inputs(sample, processor, device, min_pixels=None, max_pixels=None):
    messages = qwen_build_messages(
        sample,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    processor_kwargs = {
        "text": [text],
        "padding": True,
        "return_tensors": "pt",
    }

    if image_inputs is not None:
        processor_kwargs["images"] = image_inputs
    if video_inputs is not None:
        processor_kwargs["videos"] = video_inputs
    if video_kwargs is not None:
        processor_kwargs.update(video_kwargs)

    inputs = processor(**processor_kwargs)
    inputs = {
        k: v.to(device, non_blocking=True) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }
    return inputs, messages

@torch.inference_mode()
def call_qwen_engine_df(args, sample, model, processor):
    inputs, _ = qwen_prepare_inputs(
        sample,
        processor=processor,
        device=model.device,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    input_ids = inputs["input_ids"]

    outputs = model.generate(
        **inputs,
        do_sample=(args.temperature > 0),
        temperature=args.temperature if args.temperature > 0 else None,
        top_p=args.top_p if args.temperature > 0 else None,
        top_k=args.top_k if (args.temperature > 0 and args.top_k > 0) else None,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        use_cache=True,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    generated_ids = outputs[:, input_ids.shape[1]:]
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return response.strip()

def run_model(args, samples, model, call_model_engine_fn=None, tokenizer=None, processor=None):
    out_samples = dict()
    with torch.no_grad():
        for sample in tqdm(samples):
            response = call_model_engine_fn(args, sample, model, tokenizer, processor)

            if sample['question_type'] == 'multiple-choice':
                pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
            else:  # open question
                pred_ans = response
            out_samples[sample['id']] = pred_ans
    return out_samples

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_seed(seed_value):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)


def apply_model_patch(args, config):
    print(f"Applying patch for model_type={config.model_type}")

    if config.model_type == "qwen3_vl_moe":
        replace_qwen3_vl_moe_with_mixed_modality_forward()

    elif config.model_type == "qwen3_vl":
        if args.pooling_strategy == "Original":
            replace_qwen3_with_mixed_modality_forward()
        elif args.pooling_strategy == "Fixed":
            replace_qwen3_with_mixed_modality_forward_fixed_pooling(
                compression_rate=args.compression_rate
            )
        elif args.pooling_strategy == "DRIP":
            replace_qwen3_with_mixed_modality_forward_drip_pooling(
                compression_rate=args.compression_rate
            )
        else:
            raise ValueError("Invalid pooling_strategy")

    elif config.model_type == "qwen2_5_vl":
        replace_qwen2_5_with_mixed_modality_forward()

    else:
        replace_qwen_2_with_mixed_modality_forward()


def load_qwen_model_and_processor(args):
    processor = AutoProcessor.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)

    apply_model_patch(args, config)

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    common_kwargs = dict(
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if args.attn_implementation is not None:
        common_kwargs["attn_implementation"] = args.attn_implementation

    if config.model_type == "qwen3_vl_moe":
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            args.model_path, **common_kwargs
        )
    elif config.model_type == "qwen3_vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path, **common_kwargs
        )
    elif config.model_type == "qwen2_5_vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, **common_kwargs
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path, **common_kwargs
        )

    model.eval().cuda()
    return model, processor




import torch
import os
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from argparse import ArgumentParser

from transformers import (
    AutoProcessor,
    AutoConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

from utils.data_utils import (
    load_yaml,
    construct_prompt,
    save_json,
    process_single_sample,
    CAT_SHORT2LONG,
)
from utils.eval_utils import parse_multi_choice_response, parse_open_response
from FT_QwenVL.src.train.monkey_patch_forward import (
    replace_qwen3_with_mixed_modality_forward,
    replace_qwen2_5_with_mixed_modality_forward,
    replace_qwen_2_with_mixed_modality_forward,
    replace_qwen3_vl_moe_with_mixed_modality_forward,
    replace_qwen3_with_mixed_modality_forward_fixed_pooling,
    replace_qwen3_with_mixed_modality_forward_drip_pooling,
)


def run_model(args, samples, model, call_model_engine_fn=None, processor=None):
    out_samples = dict()
    with torch.no_grad():
        for sample in tqdm(samples):
            response = call_model_engine_fn(args, sample, model, processor)

            if sample["question_type"] == "multiple-choice":
                pred_ans = parse_multi_choice_response(
                    response,
                    sample["all_choices"],
                    sample["index2ans"],
                )
            else:
                pred_ans = parse_open_response(response)

            out_samples[sample["id"]] = pred_ans
    return out_samples


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)


def apply_model_patch(args, config):
    print(f"Applying patch for model_type={config.model_type}")

    if config.model_type == "qwen3_vl_moe":
        replace_qwen3_vl_moe_with_mixed_modality_forward()

    elif config.model_type == "qwen3_vl":
        if args.pooling_strategy == "Original":
            replace_qwen3_with_mixed_modality_forward()
        elif args.pooling_strategy == "Fixed":
            replace_qwen3_with_mixed_modality_forward_fixed_pooling(
                compression_rate=args.compression_rate
            )
        elif args.pooling_strategy == "DRIP":
            replace_qwen3_with_mixed_modality_forward_drip_pooling(
                compression_rate=args.compression_rate
            )
        else:
            raise ValueError("Invalid pooling_strategy")

    elif config.model_type == "qwen2_5_vl":
        replace_qwen2_5_with_mixed_modality_forward()

    else:
        replace_qwen_2_with_mixed_modality_forward()


def load_qwen_model_and_processor(args):
    processor = AutoProcessor.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)

    apply_model_patch(args, config)

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    common_kwargs = dict(
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if args.attn_implementation is not None:
        common_kwargs["attn_implementation"] = args.attn_implementation

    if config.model_type == "qwen3_vl_moe":
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            args.model_path, **common_kwargs
        )
    elif config.model_type == "qwen3_vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path, **common_kwargs
        )
    elif config.model_type == "qwen2_5_vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, **common_kwargs
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path, **common_kwargs
        )

    model.eval().cuda()
    return model, processor

def main():
    parser = ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="MMMU/MMMU")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--min_pixels", type=int, default=224 * 224)
    parser.add_argument("--max_pixels", type=int, default=224 * 224)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    parser.add_argument(
        "--pooling_strategy",
        type=str,
        default="Original",
        choices=["Original", "Fixed", "DRIP"],
    )
    parser.add_argument("--compression_rate", type=float, default=1.0)

    args = parser.parse_args()
    set_seed(args.seed)

    print("qwen_initializing...")

    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != "eval_params" and isinstance(value, list):
            assert len(value) == 1, f"key {key} has more than one value"
            args.config[key] = value[0]

    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values():
        sub_dataset = load_dataset(args.data_path, subject, split=args.split)
        sub_dataset_list.append(sub_dataset)

    dataset = concatenate_datasets(sub_dataset_list)

    model, processor = load_qwen_model_and_processor(args)

    samples = []
    for sample in dataset:
        sample = process_single_sample(sample)
        sample = construct_prompt(sample, args.config)

        # force short benchmark answer
        sample["final_input_prompt"] = (
            sample["final_input_prompt"].rstrip() +
            "\nAnswer with only the option letter."
        )
        # IMPORTANT:
        # for Qwen, keep raw image object/path here; processor handles it later
        samples.append(sample)

    out_samples = run_model(
        args,
        samples,
        model,
        call_model_engine_fn=call_qwen_engine_df,
        processor=processor,
    )

    save_json(args.output_path, out_samples)

    
if __name__ == "__main__":
    main()