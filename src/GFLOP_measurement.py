import argparse
import torch
import os
import json
import math
import csv
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, ProfilerActivity
from PIL import Image

from LLaVA_wrapper.llava_local.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from LLaVA_wrapper.llava_local.conversation import conv_templates
from LLaVA_wrapper.llava_local.model.builder import load_pretrained_model
from LLaVA_wrapper.llava_local.utils import disable_torch_init
from LLaVA_wrapper.llava_local.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def get_llm_config(model):
    cfg = model.config

    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config

    return cfg


def estimate_kv_cache_mb(model, seq_len, batch_size=1, bytes_per_elem=2):
    cfg = get_llm_config(model)

    num_layers = cfg.num_hidden_layers
    num_attention_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_attention_heads)
    hidden_size = cfg.hidden_size
    head_dim = hidden_size // num_attention_heads

    kv_bytes = (
        2
        * num_layers
        * batch_size
        * seq_len
        * num_kv_heads
        * head_dim
        * bytes_per_elem
    )

    return kv_bytes / 1024**2


def get_past_key_values_kv_cache_mb(outputs):
    total_bytes = 0
    for layer_cache in outputs.past_key_values:
        if isinstance(layer_cache, (tuple, list)):
            for tensor in layer_cache[:2]:
                if torch.is_tensor(tensor):
                    total_bytes += tensor.numel() * tensor.element_size()

    return total_bytes / 1024**2


class CustomDataset(Dataset):
    def __init__(
        self,
        questions,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
        conv_mode,
    ):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]

        image_file = line["image"]
        qs = line["text"]

        if self.model_config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        image = Image.open(
            os.path.join(self.image_folder, image_file)
        ).convert("RGB")

        image_tensor = process_images(
            [image],
            self.image_processor,
            self.model_config,
        )[0]

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )

        return {
            "question_id": line["question_id"],
            "image_file": image_file,
            "prompt_len": input_ids.shape[0],
            "input_ids": input_ids,
            "image_tensor": image_tensor,
            "image_size": image.size,
        }

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    assert len(batch) == 1

    item = batch[0]

    return {
        "question_id": item["question_id"],
        "image_file": item["image_file"],
        "prompt_len": item["prompt_len"],
        "input_ids": item["input_ids"].unsqueeze(0),
        "image_tensor": item["image_tensor"].unsqueeze(0),
        "image_sizes": [item["image_size"]],
    }


def create_data_loader(
    questions,
    image_folder,
    tokenizer,
    image_processor,
    model_config,
    conv_mode,
):
    dataset = CustomDataset(
        questions,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
        conv_mode,
    )

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def run_prefill_once(model, input_ids, images, image_sizes):
    return model(
        input_ids=input_ids,
        images=images,
        image_sizes=image_sizes,
        use_cache=True,
    )


def main(args):
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
    )

    model.eval()
    model.cuda()

    questions = [
        json.loads(q)
        for q in open(os.path.expanduser(args.question_file), "r")
    ]

    questions = get_chunk(
        questions,
        args.num_chunks,
        args.chunk_idx,
    )

    if args.max_samples > 0:
        questions = questions[:args.max_samples]

    data_loader = create_data_loader(
        questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        args.conv_mode,
    )

    warmup_batch = next(iter(data_loader))
    warmup_input_ids = warmup_batch["input_ids"].cuda()
    warmup_images = warmup_batch["image_tensor"].to(
        device="cuda",
        dtype=torch.float16,
    )

    for _ in range(args.warmup_iters):
        with torch.inference_mode():
            _ = run_prefill_once(
                model,
                warmup_input_ids,
                warmup_images,
                warmup_batch["image_sizes"],
            )

    torch.cuda.synchronize()

    rows = []

    tflops_list = []
    latency_ms_list = []
    peak_allocated_gb_list = []
    actual_kv_cache_mb_list = []

    for batch in tqdm(data_loader):
        input_ids = batch["input_ids"].cuda(non_blocking=True)

        images = batch["image_tensor"].to(
            device="cuda",
            dtype=torch.float16,
            non_blocking=True,
        )
        seq_len = input_ids.shape[1]


        """
            TFLOPs.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            with_flops=True,
        ) as prof:
            with torch.inference_mode():
                starter.record()
                outputs = run_prefill_once(
                    model,
                    input_ids,
                    images,
                    batch["image_sizes"],
                )
                ender.record()
        torch.cuda.synchronize()
        flops = sum(
            evt.flops or 0
            for evt in prof.key_averages()
        )
        tflops = flops / 1e12

        """
            prefill latency in milliseconds.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        with torch.inference_mode():
            outputs = run_prefill_once(
                model,
                input_ids,
                images,
                batch["image_sizes"],
            )
        ender.record()
        torch.cuda.synchronize()
        latency_ms = starter.elapsed_time(ender)

        """
            Peak allocated memory.
        """
        peak_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3


        """
            Actual KV cache memory.
        """

        actual_kv_cache_mb = get_past_key_values_kv_cache_mb(outputs)

        tflops_list.append(tflops)
        latency_ms_list.append(latency_ms)
        peak_allocated_gb_list.append(peak_allocated_gb)
        actual_kv_cache_mb_list.append(actual_kv_cache_mb)

        rows.append(
            {
                "question_id": batch["question_id"],
                "image_file": batch["image_file"],
                "prompt_len": batch["prompt_len"],
                "seq_len": seq_len,
                "prefill_tflops": tflops,
                "actual_kv_cache_mb": actual_kv_cache_mb,
                "peak_allocated_gb": peak_allocated_gb,
                "prefill_latency_ms": latency_ms,
            }
        )

    summary = {
        "model": model_name,
        "samples": len(rows),
        "avg_tflops": mean(tflops_list),
        "std_tflops": std(tflops_list),
        "min_tflops": min(tflops_list),
        "max_tflops": max(tflops_list),
        "avg_prefill_latency_ms": mean(latency_ms_list),
        "std_prefill_latency_ms": std(latency_ms_list),
        "min_prefill_latency_ms": min(latency_ms_list),
        "max_prefill_latency_ms": max(latency_ms_list),
        "avg_peak_allocated_gb": mean(peak_allocated_gb_list),
        "std_peak_allocated_gb": std(peak_allocated_gb_list),
        "min_peak_allocated_gb": min(peak_allocated_gb_list),
        "max_peak_allocated_gb": max(peak_allocated_gb_list)
    }

    summary.update(
        {
            "avg_actual_kv_cache_mb": mean(actual_kv_cache_mb_list),
            "std_actual_kv_cache_mb": std(actual_kv_cache_mb_list),
            "min_actual_kv_cache_mb": min(actual_kv_cache_mb_list),
            "max_actual_kv_cache_mb": max(actual_kv_cache_mb_list),
        }
    )


    print()
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Samples: {len(rows)}")
    print("-" * 80)
    print(f"Average TFLOPs: {summary['avg_tflops']:.2f}")
    print(f"Average Prefill Latency: {summary['avg_prefill_latency_ms']:.2f} ms")
    print(f"Average Actual KV Cache: {summary['avg_actual_kv_cache_mb']:.2f} MB")
    print(f"Average Peak Allocated Memory: {summary['avg_peak_allocated_gb']:.2f} GB")
    print("-" * 80)
    print("=" * 80)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)

    parser.add_argument(
        "--image-folder",
        type=str,
        default="/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VQAv2/test2015",
    )

    parser.add_argument(
        "--question-file",
        type=str,
        default="/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VQAv2/llava_vqav2_mscoco_test-dev2015.jsonl",
    )

    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=1)

    parser.add_argument("--warmup-iters", type=int, default=3)

    args = parser.parse_args()
    main(args)
