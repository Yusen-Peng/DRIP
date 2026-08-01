import json
import os
import traceback
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from datasets import load_dataset

from LLaVA_wrapper.llava_local.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from LLaVA_wrapper.llava_local.conversation import conv_templates, SeparatorStyle
from LLaVA_wrapper.llava_local.model.builder import load_pretrained_model
from LLaVA_wrapper.llava_local.utils import disable_torch_init
from LLaVA_wrapper.llava_local.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)


def split_list(lst, n):
    n = max(1, n)
    avg = len(lst) // n
    return [lst[i * avg:(i + 1) * avg] for i in range(n - 1)] + [lst[(n - 1) * avg:]]


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="lmms-lab/OCRBench-v2")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_folder", type=str, default="./pred_folder")
    parser.add_argument("--save_name", type=str, default="llava_ocrbench_v2")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def build_question(question):
    question = question.strip()
    question += "\nAnswer the question using a single word or phrase."
    return question


def normalize_row(row):
    """
    Convert HF dataset row into JSON-safe OCRBench-v2 prediction row.

    The PIL image itself is removed before saving.
    """
    out = {}

    for k, v in row.items():
        if k == "image":
            continue
        out[k] = v

    return out

if __name__ == "__main__":

    args = _get_args()

    output_json = os.path.join(args.output_folder, f"{args.save_name}.json")

    os.makedirs(args.output_folder, exist_ok=True)

    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))

    print("torch.cuda.device_count() =", torch.cuda.device_count())

    device = "cuda:0"

    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)

    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(

        model_path=model_path,

        model_base=args.model_base,

        model_name=model_name,

        device=device,

    )

    conv_mode = args.conv_mode

    dataset = load_dataset(

        args.dataset_path,

        split=args.dataset_split,

        cache_dir=args.cache_dir,

    )

    if args.resume and os.path.exists(output_json):

        results = load_json(output_json)

        start_idx = len(results)

        print(f"[resume] found {start_idx} existing predictions")

    else:

        results = []

        start_idx = 0

    for idx in tqdm(range(start_idx, len(dataset)), desc="ocrbench-v2"):

        row = dataset[idx]

        image = row["image"].convert("RGB")

        qs = build_question(row["question"])

        if model.config.mm_use_im_start_end:

            qs = (

                DEFAULT_IM_START_TOKEN

                + DEFAULT_IMAGE_TOKEN

                + DEFAULT_IM_END_TOKEN

                + "\n"

                + qs

            )

        else:

            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[conv_mode].copy()

        conv.append_message(conv.roles[0], qs)

        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        image_tensor = process_images(

            [image],

            image_processor,

            model.config,

        )

        input_ids = tokenizer_image_token(

            prompt,

            tokenizer,

            IMAGE_TOKEN_INDEX,

            return_tensors="pt",

        ).unsqueeze(0).to(device=device, non_blocking=True)

        stop_str = (

            conv_templates[conv_mode].sep

            if conv_templates[conv_mode].sep_style != SeparatorStyle.TWO

            else conv_templates[conv_mode].sep2

        )

        with torch.inference_mode():
            # output_ids = model.generate(
            #     input_ids,
            #     images=image_tensor.to(dtype=torch.float16, device=device,non_blocking=True),
            #     do_sample=args.temperature > 0,
            #     temperature=args.temperature,
            #     top_p=args.top_p,
            #     num_beams=args.num_beams,
            #     max_new_tokens=args.max_new_tokens,
            #     use_cache=True,
            # )

            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
            stop_ids = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
                tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            ]
            # stop_ids += newline_ids
            stop_ids = [x for x in stop_ids if x is not None and x != tokenizer.unk_token_id]
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                use_cache=True
            )

        input_token_len = input_ids.shape[1]

        if output_ids.shape[1] >= input_token_len:

            gen_ids = output_ids[:, input_token_len:]

        else:

            gen_ids = output_ids

        outputs = tokenizer.batch_decode(

            gen_ids,

            skip_special_tokens=True,

        )[0].strip()

        if outputs.endswith(stop_str):

            outputs = outputs[:-len(stop_str)].strip()

        result_row = normalize_row(row)

        result_row["predict"] = outputs

        results.append(result_row)

        # save every 50 samples so crashes can resume

        if len(results) % 50 == 0:

            save_json(results, output_json)

        del image, image_tensor, input_ids, output_ids

        torch.cuda.empty_cache()

    save_json(results, output_json)

    print(f"[saved] {output_json}")
