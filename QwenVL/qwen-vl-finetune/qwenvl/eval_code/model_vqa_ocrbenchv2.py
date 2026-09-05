import json
import os
import traceback
from argparse import ArgumentParser

import torch
from tqdm import tqdm
from datasets import load_dataset
import sys
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../"))
sys.path.insert(0, PROJECT_ROOT)

from model.qwen3vl_compressed import CompressedQwen3VLForConditionalGeneration



def split_list(lst, n):
    n = max(1, n)
    avg = len(lst) // n
    return (
        [lst[i * avg:(i + 1) * avg] for i in range(n - 1)]
        + [lst[(n - 1) * avg:]]
    )


def save_json(obj, path):
    output_dir = os.path.dirname(path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=4,
            ensure_ascii=False,
        )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="lmms-lab/OCRBench-v2",
    )

    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="./pred_folder",
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default="qwen3vl_ocrbench_v2",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model_base",
        type=str,
        default=None,
    )

    # Kept for interface compatibility.
    parser.add_argument(
        "--conv_mode",
        type=str,
        default="qwen3_vl",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )


    parser.add_argument(
        "--merge-strategy",
        type=str,
        default="none",
    )

    parser.add_argument(
        "--compression-rate",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--sampling-temperature",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--drip-path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--mlp-ratio",
        type=float,
        default=1.0,
    )

    return parser.parse_args()


def build_question(question):
    question = question.strip()

    question += (
        "\nAnswer the question using a single word or phrase."
    )

    return question


def normalize_row(row):
    """
    Convert HF dataset row into JSON-safe OCRBench-v2
    prediction row.

    The PIL image itself is removed before saving.
    """
    out = {}

    for k, v in row.items():
        if k == "image":
            continue

        out[k] = v

    return out


def load_model(args, device):
    model_path = os.path.expanduser(
        args.model_path
    )

    if args.model_base is not None:
        # ----------------------------------------------------
        # LoRA checkpoint
        # ----------------------------------------------------

        model_base = os.path.expanduser(
            args.model_base
        )

        print(
            f"Loading base model from {model_base}"
        )

        if args.merge_strategy == "Fixed":
            print(f"🌊 Loading compressed Qwen3VL: Fixed, rate={args.compression_rate}")
            model = CompressedQwen3VLForConditionalGeneration.from_pretrained(
                    model_base,
                    attn_implementation="flash_attention_2",
                    dtype=torch.bfloat16,
                    device_map={"": device},
            )
            model.model.set_compressor(merge_strategy=args.merge_strategy, compression_rate=args.compression_rate, temperature=args.sampling_temperature)

        elif args.merge_strategy == "DRIP":
            print(f"🌊 Loading compressed Qwen3VL: DRIP, rate={args.compression_rate}, temperature={args.sampling_temperature}")
            model = CompressedQwen3VLForConditionalGeneration.from_pretrained(
                    model_base,
                    attn_implementation="flash_attention_2",
                    dtype=torch.bfloat16,
                    device_map="auto"
            )
            model.model.set_compressor(merge_strategy=args.merge_strategy, compression_rate=args.compression_rate, temperature=args.sampling_temperature, drip_path=args.drip_path, mlp_ratio=args.mlp_ratio)

        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_base,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map={"": device},
            )

        print(
            f"Loading LoRA adapter from {model_path}"
        )

        model = PeftModel.from_pretrained(
            model,
            model_path,
        )

        # Merge LoRA weights for inference.
        model = model.merge_and_unload()

        processor = AutoProcessor.from_pretrained(
            model_base
        )

        model_name = os.path.basename(
            os.path.normpath(model_path)
        )

    else:
        # ----------------------------------------------------
        # Full checkpoint
        # ----------------------------------------------------

        print(
            f"Loading model from {model_path}"
        )

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": device},
        )

        processor = AutoProcessor.from_pretrained(
            model_path
        )

        model_name = os.path.basename(
            os.path.normpath(model_path)
        )

    model.eval()

    return model, processor, model_name


def generate_answer(
    model,
    processor,
    image,
    question,
    args,
    device,
):
    qs = build_question(question)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": qs,
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "use_cache": True,
    }

    if args.temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
            }
        )

        if args.top_p is not None:
            generation_kwargs["top_p"] = args.top_p

    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )

    # Qwen generate() returns:
    #
    # [prompt tokens | generated tokens]
    #
    # Remove prompt tokens before decoding.
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            inputs.input_ids,
            generated_ids,
        )
    ]

    outputs = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return outputs


def eval_worker(
    args,
    data,
    eval_id,
    output_queue,
):
    """
    Optional multi-GPU worker implementation.

    Currently the main evaluation path below remains single GPU,
    matching the original OCRBench-v2 wrapper.
    """
    try:
        print(
            f"[worker {eval_id}] start, "
            f"samples={len(data)}"
        )

        device = f"cuda:{eval_id}"

        torch.cuda.set_device(eval_id)

        model, processor, model_name = load_model(
            args,
            device,
        )

        output_data = []

        for i in tqdm(
            range(len(data)),
            desc=f"worker-{eval_id}",
        ):
            row = data[i]

            if row.get(
                "predict",
                None,
            ) not in [None, "", 0]:

                output_data.append(
                    normalize_row(row)
                )

                continue

            image = row["image"].convert(
                "RGB"
            )

            outputs = generate_answer(
                model=model,
                processor=processor,
                image=image,
                question=row["question"],
                args=args,
                device=device,
            )

            result_row = normalize_row(
                row
            )

            result_row["predict"] = outputs

            output_data.append(
                result_row
            )

        output_queue.put(
            {
                eval_id: output_data
            }
        )

        print(
            f"[worker {eval_id}] completed"
        )

    except Exception:
        print(
            f"🔥 worker {eval_id} crashed"
        )

        traceback.print_exc()

        raise


if __name__ == "__main__":
    args = _get_args()

    output_json = os.path.join(
        args.output_folder,
        f"{args.save_name}.json",
    )

    os.makedirs(
        args.output_folder,
        exist_ok=True,
    )

    print(
        "CUDA_VISIBLE_DEVICES =",
        os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ),
    )

    print(
        "torch.cuda.device_count() =",
        torch.cuda.device_count(),
    )

    # --------------------------------------------------------
    # Single-GPU evaluation
    # --------------------------------------------------------

    device = "cuda:0"

    torch.cuda.set_device(0)

    model, processor, model_name = load_model(
        args,
        device,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = load_dataset(
        args.dataset_path,
        split=args.dataset_split,
        cache_dir=args.cache_dir,
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    if (
        args.resume
        and os.path.exists(output_json)
    ):
        results = load_json(
            output_json
        )

        start_idx = len(results)

        print(
            f"[resume] found "
            f"{start_idx} existing predictions"
        )

    else:
        results = []
        start_idx = 0

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    for idx in tqdm(
        range(
            start_idx,
            len(dataset),
        ),
        desc="ocrbench-v2",
    ):
        row = dataset[idx]

        image = row["image"].convert(
            "RGB"
        )

        outputs = generate_answer(
            model=model,
            processor=processor,
            image=image,
            question=row["question"],
            args=args,
            device=device,
        )

        result_row = normalize_row(
            row
        )

        result_row["predict"] = outputs

        results.append(
            result_row
        )

        # Save every 50 samples so crashes can resume.
        if len(results) % 50 == 0:
            save_json(
                results,
                output_json,
            )

        del image

        # Not strictly necessary every iteration,
        # but kept close to your existing implementation.
        torch.cuda.empty_cache()

    save_json(
        results,
        output_json,
    )

    print(
        f"[saved] {output_json}"
    )
