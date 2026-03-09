import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid
import math
import random
import numpy as np
import base64
from io import BytesIO

from transformers import (
    AutoProcessor,
    AutoConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info
from PIL import Image


all_options = ["A", "B", "C", "D"]


def set_seed(seed_value):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.lower() in ["nan", "none"]:
        return True
    return False


def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image))).convert("RGB")


def load_qwen_model_and_processor(args):
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

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


def build_question(row, options, lang="en", single_pred_prompt=False):
    question = row["question"]
    hint = row["hint"] if "hint" in row else None

    if not is_none(hint):
        question = hint + "\n" + question

    for option_char, option in zip(all_options[:len(options)], options):
        question += f"\n{option_char}. {option}"

    if single_pred_prompt:
        if lang == "cn":
            question += "\n请直接回答选项字母。"
        else:
            question += "\nAnswer with only the option letter."

    return question


def build_qwen_messages(image, question, min_pixels=None, max_pixels=None):
    image_item = {
        "type": "image",
        "image": image,
    }
    if min_pixels is not None:
        image_item["min_pixels"] = min_pixels
    if max_pixels is not None:
        image_item["max_pixels"] = max_pixels

    messages = [
        {
            "role": "user",
            "content": [
                image_item,
                {"type": "text", "text": question},
            ],
        }
    ]
    return messages


def prepare_qwen_inputs(messages, processor, device):
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
    return inputs


@torch.inference_mode()
def generate_response(args, model, processor, image, question):
    messages = build_qwen_messages(
        image=image,
        question=question,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    inputs = prepare_qwen_inputs(messages, processor, model.device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": args.temperature > 0,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "pad_token_id": processor.tokenizer.pad_token_id,
    }

    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            generation_kwargs["top_p"] = args.top_p
        if args.top_k is not None and args.top_k > 0:
            generation_kwargs["top_k"] = args.top_k

    outputs = model.generate(**inputs, **generation_kwargs)
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]

    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return response.strip()


def eval_model(args):
    set_seed(args.seed)

    model, processor = load_qwen_model_and_processor(args)

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    with open(answers_file, "w") as ans_file:
        for _, row in tqdm(questions.iterrows(), total=len(questions)):
            options = get_options(row, all_options)
            cur_option_char = all_options[:len(options)]

            num_rounds = len(options) if args.all_rounds else 1

            for round_idx in range(num_rounds):
                idx = row["index"]
                image = load_image_from_base64(row["image"])

                question = build_question(
                    row=row,
                    options=options,
                    lang=args.lang,
                    single_pred_prompt=args.single_pred_prompt,
                )
                cur_prompt = question

                response = generate_response(
                    args=args,
                    model=model,
                    processor=processor,
                    image=image,
                    question=question,
                )

                ans_id = shortuuid.uuid()
                ans_file.write(
                    json.dumps(
                        {
                            "question_id": idx,
                            "round_id": round_idx,
                            "prompt": cur_prompt,
                            "text": response,
                            "options": options,
                            "option_char": cur_option_char,
                            "answer_id": ans_id,
                            "model_id": args.model_path,
                            "metadata": {},
                        }
                    )
                    + "\n"
                )
                ans_file.flush()

                # rotate options for all-rounds mode
                options = options[1:] + options[:1]
                cur_option_char = cur_option_char[1:] + cur_option_char[:1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
    )

    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--min_pixels", type=int, default=224 * 224)
    parser.add_argument("--max_pixels", type=int, default=224 * 224)

    args = parser.parse_args()
    eval_model(args)