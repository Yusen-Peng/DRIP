import argparse
import torch
import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import shortuuid
import math
from PIL import Image

from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [
        lst[i:i + chunk_size]
        for i in range(0, len(lst), chunk_size)
    ]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(
        self,
        questions,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
    ):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]

        image_file = line["image"]
        qs = line["text"]

        image_path = os.path.join(
            self.image_folder,
            image_file,
        )

        image = Image.open(image_path).convert("RGB")

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

        # Qwen processor handles:
        #   1. chat template
        #   2. tokenization
        #   3. image preprocessing
        #   4. multimodal position/grid metadata
        inputs = self.image_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        return inputs, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    # We intentionally enforce batch_size == 1 below.
    #
    # Qwen's BatchFeature already contains a batch dimension, so
    # stacking it again would incorrectly create an extra dimension.
    inputs, image_sizes = batch[0]

    return inputs, [image_sizes]


# DataLoader
def create_data_loader(
    questions,
    image_folder,
    tokenizer,
    image_processor,
    model_config,
    batch_size=1,
    num_workers=4,
):
    assert batch_size == 1, "batch_size must be 1"

    dataset = CustomDataset(
        questions,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return data_loader



def load_model(args):
    model_path = os.path.expanduser(args.model_path)
    if args.model_base is not None:
        # LoRA checkpoint
        model_base = os.path.expanduser(args.model_base)

        print(f"Loading base model from {model_base}")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_base,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        print(f"Loading LoRA adapter from {model_path}")

        model = PeftModel.from_pretrained(
            model,
            model_path,
        )

        # Merge LoRA into the base weights for inference.
        model = model.merge_and_unload()

        processor = AutoProcessor.from_pretrained(
            model_base,
        )

        model_name = os.path.basename(
            os.path.normpath(model_path)
        )

    else:
        # Normal full checkpoint
        print(f"Loading model from {model_path}")

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        processor = AutoProcessor.from_pretrained(
            model_path,
        )

        model_name = os.path.basename(
            os.path.normpath(model_path)
        )
    model.eval()
    return model, processor, model_name


def eval_model(args):
    model, processor, model_name = load_model(args)

    # Keep names similar to LLaVA.
    tokenizer = processor.tokenizer
    image_processor = processor

    questions = [
        json.loads(q)
        for q in open(
            os.path.expanduser(args.question_file),
            "r",
        )
    ]

    questions = get_chunk(
        questions,
        args.num_chunks,
        args.chunk_idx,
    )

    answers_file = os.path.expanduser(args.answers_file)

    answer_dir = os.path.dirname(answers_file)

    if answer_dir:
        os.makedirs(
            answer_dir,
            exist_ok=True,
        )

    ans_file = open(
        answers_file,
        "w",
    )

    data_loader = create_data_loader(
        questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
    )

    for (inputs, image_sizes), line in tqdm(
        zip(data_loader, questions),
        total=len(questions),
    ):
        idx = line["question_id"]
        cur_prompt = line["text"]
        inputs = inputs.to(model.device)

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

        # model.generate() returns:
        # [prompt tokens | newly generated tokens]
        # Remove the prompt portion before decoding.
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

        ans_id = shortuuid.uuid()

        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": cur_prompt,
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                }
            )
            + "\n"
        )
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
    )

    # Kept for interface compatibility with LLaVA scripts.
    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--image-folder",
        type=str,
        default="",
    )

    parser.add_argument(
        "--question-file",
        type=str,
        default="tables/question.jsonl",
    )

    parser.add_argument(
        "--answers-file",
        type=str,
        default="answer.jsonl",
    )

    parser.add_argument(
        "--conv-mode",
        type=str,
        default="qwen3_vl",
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
        "--temperature",
        type=float,
        default=0.2,
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

    args = parser.parse_args()

    eval_model(args)

