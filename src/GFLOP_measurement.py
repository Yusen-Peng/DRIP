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
    with torch.inference_mode():
        _ = model(
            input_ids=warmup_batch["input_ids"].cuda(),
            images=warmup_batch["image_tensor"].cuda().half(),
            image_sizes=warmup_batch["image_sizes"],
            use_cache=True,
        )

    torch.cuda.synchronize()
    rows = []
    tflops_list = []

    for batch in tqdm(data_loader):
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        images = batch["image_tensor"].to(
            device="cuda",
            dtype=torch.float16,
            non_blocking=True,
        )

        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            with_flops=True,
        ) as prof:

            with torch.inference_mode():
                
                # We only run the prefill (i.e., forward pass with use_cache=True) to measure the GFLOPs for the prefill stage.
                # no decoding is performed, so the generation loop is not included in the measurement.
                _ = model(
                    input_ids=input_ids,
                    images=images,
                    image_sizes=batch["image_sizes"],
                    use_cache=True,
                )

        torch.cuda.synchronize()

        flops = sum(
            evt.flops or 0
            for evt in prof.key_averages()
        )

        tflops = flops / 1e12

        tflops_list.append(tflops)

        rows.append(
            {
                "question_id": batch["question_id"],
                "image_file": batch["image_file"],
                "prompt_len": batch["prompt_len"],
                "prefill_tflops": tflops,
            }
        )
    avg_tflops = sum(tflops_list) / len(tflops_list)

    std_tflops = (
        sum(
            (x - avg_tflops) ** 2
            for x in tflops_list
        )
        / len(tflops_list)
    ) ** 0.5

    print()
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Samples: {len(tflops_list)}")
    print(f"Average TFLOPs: {avg_tflops:.2f}")
    print(f"Std TFLOPs: {std_tflops:.2f}")
    print(f"Min TFLOPs: {min(tflops_list):.2f}")
    print(f"Max TFLOPs: {max(tflops_list):.2f}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VQAv2/test2015")
    parser.add_argument("--question-file", type=str, default="/fs/scratch/PAS2836/yusenpeng_dataset/LLaVA_eval/VQAv2/llava_vqav2_mscoco_test-dev2015.jsonl")
    parser.add_argument(
        "--conv-mode",
        type=str,
        default="llava_v1",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--chunk-idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    main(args)
