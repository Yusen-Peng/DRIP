import argparse
import json
import math
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
    return [
        lst[i:i + chunk_size]
        for i in range(0, len(lst), chunk_size)
    ]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class SyntheticOCRDataset(Dataset):

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

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):

        sample = self.questions[index]

        image_file = sample["image"]
        question = sample["question"]

        if self.model_config.mm_use_im_start_end:
            question = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + question
            )
        else:
            question = DEFAULT_IMAGE_TOKEN + "\n" + question

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        image_path = os.path.join(
            self.image_folder,
            image_file,
        )

        image = Image.open(image_path).convert("RGB")

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

        return (
            input_ids,
            image_tensor,
            image.size,
        )


def collate_fn(batch):

    input_ids, image_tensors, image_sizes = zip(*batch)

    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)

    return input_ids, image_tensors, image_sizes


def create_data_loader(
    questions,
    image_folder,
    tokenizer,
    image_processor,
    model_config,
    conv_mode,
    batch_size=1,
    num_workers=4,
):

    assert batch_size == 1

    dataset = SyntheticOCRDataset(
        questions=questions,
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        model_config=model_config,
        conv_mode=conv_mode,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )


def eval_model(args):

    disable_torch_init()

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = (
        load_pretrained_model(
            model_path,
            args.model_base,
            model_name,
        )
    )

    # ---------------------------------------------------------
    # Load synthetic questions
    # ---------------------------------------------------------

    with open(args.question_file, "r") as f:
        questions = [
            json.loads(line)
            for line in f
        ]

    questions = get_chunk(
        questions,
        args.num_chunks,
        args.chunk_idx,
    )

    print(f"Loaded {len(questions)} synthetic QA samples")

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    answers_file = os.path.expanduser(args.answers_file)

    os.makedirs(
        os.path.dirname(answers_file),
        exist_ok=True,
    )

    ans_file = open(answers_file, "w")

    # ---------------------------------------------------------
    # Conversation mode
    # ---------------------------------------------------------

    if (
        "plain" in model_name
        and "finetune" not in model_name.lower()
        and "mmtag" not in args.conv_mode
    ):
        args.conv_mode += "_mmtag"

    # ---------------------------------------------------------
    # Data loader
    # ---------------------------------------------------------

    data_loader = create_data_loader(
        questions=questions,
        image_folder=args.image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        model_config=model.config,
        conv_mode=args.conv_mode,
        num_workers=args.num_workers,
    )

    # ---------------------------------------------------------
    # Inference
    # ---------------------------------------------------------

    for (
        input_ids,
        image_tensor,
        image_sizes,
    ), sample in tqdm(
        zip(data_loader, questions),
        total=len(questions),
    ):

        input_ids = input_ids.to(
            device="cuda",
            non_blocking=True,
        )

        image_tensor = image_tensor.to(
            dtype=torch.float16,
            device="cuda",
            non_blocking=True,
        )

        with torch.inference_mode():

            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        prediction = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )[0].strip()

        result = {
            "id": sample["id"],
            "image": sample["image"],
            "question": sample["question"],
            "prediction": prediction,
            "answer": sample["answer"],
            "row": sample["row"],
            "column": sample["column"],
            "model_id": model_name,
        }

        ans_file.write(
            json.dumps(result) + "\n"
        )

        # Useful for long jobs / interrupted jobs
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--image-folder",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--question-file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--answers-file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--conv-mode",
        type=str,
        default="vicuna_v1",
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

    # We only want one character.
    # Give it a little room in case model produces prose.
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    eval_model(args)