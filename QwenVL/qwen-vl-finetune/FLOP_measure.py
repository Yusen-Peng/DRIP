import argparse
import json
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# ============================================================
# Local imports
# ============================================================

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FILE_DIR)

from qwenvl.model.qwen3vl_compressed import (
    CompressedQwen3VLForConditionalGeneration,
)


# ============================================================
# Utilities
# ============================================================

def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def get_kv_cache_mb(outputs):
    """Return actual memory occupied by K/V tensors."""
    cache = outputs.past_key_values

    if cache is None:
        return 0.0

    total_bytes = 0

    if isinstance(cache, (tuple, list)):
        for layer_cache in cache:
            if isinstance(layer_cache, (tuple, list)):
                for tensor in layer_cache[:2]:
                    if torch.is_tensor(tensor):
                        total_bytes += tensor.numel() * tensor.element_size()

    elif hasattr(cache, "layers"):
        for layer in cache.layers:
            for name in ["keys", "values"]:
                tensor = getattr(layer, name, None)
                if torch.is_tensor(tensor):
                    total_bytes += tensor.numel() * tensor.element_size()

    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for tensor in cache.key_cache:
            if torch.is_tensor(tensor):
                total_bytes += tensor.numel() * tensor.element_size()

        for tensor in cache.value_cache:
            if torch.is_tensor(tensor):
                total_bytes += tensor.numel() * tensor.element_size()

    else:
        raise RuntimeError(f"Unknown cache type: {type(cache)}")

    return total_bytes / 1024**2


# ============================================================
# Dataset
# ============================================================

class ImageFolderDataset(Dataset):

    def __init__(
        self,
        image_folder,
        processor,
        prompt="Describe this image.",
    ):
        self.image_folder = image_folder
        self.processor = processor
        self.prompt = prompt

        valid_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
            ".tif",
            ".tiff",
        }

        self.image_files = sorted([
            f
            for f in os.listdir(image_folder)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ])

        if len(self.image_files) == 0:
            raise RuntimeError(
                f"No images found in {image_folder}"
            )

        print(
            f"📄 Found {len(self.image_files)} images "
            f"in {image_folder}"
        )

    def __len__(self):
        return len(self.image_files)


    def __getitem__(self, index):

        image_file = self.image_files[index]

        image_path = os.path.join(
            self.image_folder,
            image_file,
        )

        image = Image.open(
            image_path
        ).convert("RGB")

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
                        "text": self.prompt,
                    },
                ],
            }
        ]

        # EXACT same preprocessing path as evaluation
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        return {
            "image_file": image_file,
            **inputs,
        }




def collate_fn(batch):
    assert len(batch) == 1
    return batch[0]


# ============================================================
# Model
# ============================================================

def load_model(args):

    if args.merge_strategy.lower() == "none":

        print("🌊 Loading vanilla Qwen3-VL")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,
        )

    else:

        print(
            f"🌊 Loading compressed Qwen3-VL\n"
            f"   strategy = {args.merge_strategy}\n"
            f"   compression rate = {args.compression_rate}"
        )

        model = (
            CompressedQwen3VLForConditionalGeneration
            .from_pretrained(
                args.model_path,
                dtype=torch.bfloat16,
                attn_implementation=args.attn_implementation,
            )
        )

        model.model.set_compressor(
            merge_strategy=args.merge_strategy,
            compression_rate=args.compression_rate,
            temperature=args.temperature,
            drip_path=args.drip_path,
            mlp_ratio=args.mlp_ratio,
        )

    model.eval()
    model.cuda()

    return model


# ============================================================
# Forward
# ============================================================

def move_to_cuda(batch, model_dtype):

    output = {}

    for key, value in batch.items():

        if not torch.is_tensor(value):
            output[key] = value
            continue

        if key == "pixel_values":
            output[key] = value.to(
                "cuda",
                dtype=model_dtype,
                non_blocking=True,
            )
        else:
            output[key] = value.to(
                "cuda",
                non_blocking=True,
            )

    return output


def run_prefill(model, batch):

    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        pixel_values=batch.get("pixel_values"),
        image_grid_thw=batch.get("image_grid_thw"),
        use_cache=True,
    )


# ============================================================
# Main
# ============================================================

def main(args):

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")

    model = load_model(args)

    model_dtype = next(model.parameters()).dtype

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = ImageFolderDataset(
        image_folder=args.image_folder,
        processor=processor,
        prompt=args.prompt,
    )

    if args.max_samples > 0:
        dataset.image_files = dataset.image_files[
            :args.max_samples
        ]

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Profiling {len(dataset)} images")

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    warmup_batch = next(iter(data_loader))

    warmup_batch = move_to_cuda(
        warmup_batch,
        model_dtype,
    )

    print(
        f"Warming up for "
        f"{args.warmup_iters} iterations..."
    )

    for _ in range(args.warmup_iters):
        with torch.inference_mode():
            _ = run_prefill(
                model,
                warmup_batch,
            )

    torch.cuda.synchronize()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    rows = []

    for batch in tqdm(data_loader):

        batch = move_to_cuda(
            batch,
            model_dtype,
        )

        input_ids = batch["input_ids"]

        original_seq_len = input_ids.shape[1]

        image_token_id = model.config.image_token_id

        visual_tokens_before = int(
            (input_ids == image_token_id)
            .sum()
            .item()
        )

        # ----------------------------------------------------
        # Qwen image geometry
        # ----------------------------------------------------

        grid_thw = batch["image_grid_thw"][0]

        t, h, w = map(
            int,
            grid_thw.tolist(),
        )

        native_vit_patches = t * h * w

        merge_size = (
            processor.image_processor.merge_size
        )

        expected_visual_tokens = (
            native_vit_patches
            // (merge_size ** 2)
        )

        print(
            f"\n📄 {batch['image_file']}"
            f"\n   grid = {t} x {h} x {w}"
            f"\n   ViT patches = {native_vit_patches}"
            f"\n   visual tokens = {visual_tokens_before}"
            f"\n   expected visual tokens = {expected_visual_tokens}"
            f"\n   original sequence = {original_seq_len}"
        )

        # ====================================================
        # FLOPs
        # ====================================================

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            with_flops=True,
        ) as prof:

            with torch.inference_mode():
                outputs = run_prefill(
                    model,
                    batch,
                )

        torch.cuda.synchronize()

        flops = sum(
            event.flops or 0
            for event in prof.key_averages()
        )

        tflops = flops / 1e12

        # ====================================================
        # Latency + peak memory
        # ====================================================

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = torch.cuda.Event(
            enable_timing=True
        )
        end = torch.cuda.Event(
            enable_timing=True
        )

        start.record()

        with torch.inference_mode():
            outputs = run_prefill(
                model,
                batch,
            )

        end.record()

        torch.cuda.synchronize()

        latency_ms = start.elapsed_time(end)

        peak_memory_gb = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        # ====================================================
        # KV cache
        # ====================================================

        kv_cache_mb = get_kv_cache_mb(
            outputs
        )

        # ====================================================
        # Post-compression sequence
        # ====================================================

        if args.merge_strategy.lower() == "none":

            llm_seq_len = original_seq_len
            visual_tokens_after = visual_tokens_before

        else:

            cache = outputs.past_key_values

            llm_seq_len = int(
                cache.get_seq_length()
            )

            text_tokens = (
                original_seq_len
                - visual_tokens_before
            )

            visual_tokens_after = (
                llm_seq_len
                - text_tokens
            )

        print(
            f"   after compression = "
            f"{visual_tokens_after} visual tokens"
            f"\n   LLM sequence = {llm_seq_len}"
            f"\n   TFLOPs = {tflops:.3f}"
            f"\n   latency = {latency_ms:.2f} ms"
        )

        rows.append({
            "image_file":
                batch["image_file"],

            "grid_thw":
                [t, h, w],

            "native_vit_patches":
                native_vit_patches,

            "visual_tokens_before":
                visual_tokens_before,

            "visual_tokens_after":
                visual_tokens_after,

            "original_seq_len":
                original_seq_len,

            "llm_seq_len":
                llm_seq_len,

            "prefill_tflops":
                tflops,

            "prefill_latency_ms":
                latency_ms,

            "kv_cache_mb":
                kv_cache_mb,

            "peak_memory_gb":
                peak_memory_gb,
        })

    # ========================================================
    # Summary
    # ========================================================

    def values(key):
        return [row[key] for row in rows]

    summary = {
        "model":
            args.model_path,

        "strategy":
            args.merge_strategy,

        "compression_rate":
            1.0
            if args.merge_strategy.lower() == "none"
            else args.compression_rate,

        "samples":
            len(rows),

        "avg_visual_tokens_before":
            mean(values("visual_tokens_before")),

        "avg_visual_tokens_after":
            mean(values("visual_tokens_after")),

        "avg_llm_seq_len":
            mean(values("llm_seq_len")),

        "avg_tflops":
            mean(values("prefill_tflops")),

        "std_tflops":
            std(values("prefill_tflops")),

        "avg_latency_ms":
            mean(values("prefill_latency_ms")),

        "avg_kv_cache_mb":
            mean(values("kv_cache_mb")),

        "avg_peak_memory_gb":
            mean(values("peak_memory_gb")),
    }

    print("\n" + "=" * 80)

    print(
        f"Strategy: {args.merge_strategy}"
    )

    print(
        f"Samples: {len(rows)}"
    )

    print(
        f"Visual tokens: "
        f"{summary['avg_visual_tokens_before']:.2f}"
        f" -> "
        f"{summary['avg_visual_tokens_after']:.2f}"
    )

    print(
        f"Average LLM sequence: "
        f"{summary['avg_llm_seq_len']:.2f}"
    )

    print(
        f"Average TFLOPs: "
        f"{summary['avg_tflops']:.3f} "
        f"± {summary['std_tflops']:.3f}"
    )

    print(
        f"Average latency: "
        f"{summary['avg_latency_ms']:.2f} ms"
    )

    print(
        f"Average KV cache: "
        f"{summary['avg_kv_cache_mb']:.2f} MB"
    )

    print(
        f"Average peak memory: "
        f"{summary['avg_peak_memory_gb']:.2f} GB"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if args.output_file:

        os.makedirs(
            os.path.dirname(args.output_file)
            or ".",
            exist_ok=True,
        )

        with open(
            args.output_file,
            "w",
        ) as f:
            json.dump(
                {
                    "summary": summary,
                    "samples": rows,
                },
                f,
                indent=4,
            )

        print(
            f"💾 Saved to {args.output_file}"
        )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen3-VL-4B-Instruct",
    )

    parser.add_argument(
        "--image-folder",
        required=True,
    )

    parser.add_argument(
        "--prompt",
        default="Describe this image.",
    )

    parser.add_argument(
        "--merge-strategy",
        default="none",
    )

    parser.add_argument(
        "--compression-rate",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--mlp-ratio",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--drip-path",
        default=None,
    )

    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--output-file",
        default=None,
    )

    args = parser.parse_args()

    main(args)


"""
python FLOP_measure.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_10 \
    --merge-strategy none \
    --image-folder /users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune/img4flop

python FLOP_measure.py \
    --model-path /fs/scratch/PAS2836/yusenpeng_checkpoint/Qwen3VL_SFT_Fixed_4x_NEW_PIPELINE_transfer \
    --merge-strategy fixed \
    --image-folder /users/PAS2912/yusenpeng/DRIP/QwenVL/qwen-vl-finetune/img4flop
"""
