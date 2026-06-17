import torch
import os
import random

import numpy as np
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from LLaVA_wrapper.llava_local.model.builder import load_pretrained_model
from LLaVA_wrapper.llava_local.mm_utils import get_model_name_from_path

from argparse import ArgumentParser

from LLaVA_wrapper.llava_local.mmmu_utils.data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG
from LLaVA_wrapper.llava_local.mmmu_utils.model_utils import llava_image_processor
from LLaVA_wrapper.llava_local.mmmu_utils.eval_utils import parse_multi_choice_response


def call_llava_engine_df(args, sample, model, tokenizer=None, processor=None):
    from src.LLaVA_wrapper.llava_local.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from src.LLaVA_wrapper.llava_local.conversation import conv_templates, SeparatorStyle

    def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def deal_with_prompt(input_text, mm_use_im_start_end):
        qs = input_text
        if mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        return qs

    prompt = sample['final_input_prompt']
    prompt = deal_with_prompt(prompt, model.config.mm_use_im_start_end)
    conv = conv_templates['qwen_v2'].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    image = sample['image']
    if image is not None:
        # output_ids = model.generate(
        #     input_ids,
        #     images=image.unsqueeze(0).half().cuda(),
        #     do_sample=True,
        #     temperature=1,
        #     top_p=None,
        #     num_beams=5,
        #     max_new_tokens=128,
        #     use_cache=True)
    
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        stop_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]
        # stop_ids += newline_ids
        stop_ids = [x for x in stop_ids if x is not None and x != tokenizer.unk_token_id]
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image.unsqueeze(0).half().cuda(),
                do_sample=True,
                temperature=1,
                top_p=None,
                num_beams=5,
                max_new_tokens=128,
                eos_token_id=stop_ids,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                use_cache=True
            )


        input_token_len = input_ids.shape[1]
        # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        # if n_diff_input_output > 0:
        #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        # response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]



        # Some generate() paths return [prompt + generated tokens],
        # while others return only generated tokens.
        if output_ids.shape[1] >= input_token_len:
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
            response_ids = output_ids[:, input_token_len:]
        else:
            # print(
            #     f"[Warning] output_ids shorter than input_ids: "
            #     f"output_ids={output_ids.shape}, input_ids={input_ids.shape}. "
            #     f"Assuming output_ids contains generated tokens only."
            # )
            response_ids = output_ids
        response = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]    
    
    else:  # multiple images actually
        if sample['question_type'] == 'multiple-choice':
            all_choices = sample['all_choices']
            response = random.choice(all_choices)
        else:
            response = 'INVALID GENERATION FOR MULTIPLE IMAGE INPUTS'

    return response





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

def main():
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='llava1.5_13b_val.json',
                        help='name of saved json')
    parser.add_argument('--config_path', type=str, default="configs/llava1.5.yaml")
    parser.add_argument('--data_path', type=str, default="MMMU/MMMU") # hf dataset path.
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    print('llava_initializing...')
    processor = None
    call_model_engine = call_llava_engine_df
    vis_process_func = llava_image_processor

    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]

    # run for each subject
    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values():
        sub_dataset = load_dataset(args.data_path, subject, split=args.split)
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)


    # load model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, vis_processors, _ = load_pretrained_model(args.model_path, args.model_base,
                                                                model_name)

    samples = []
    for sample in dataset:
        sample = process_single_sample(sample)

        sample = construct_prompt(sample, args.config)
        if sample['image']:
            sample['image'] = vis_process_func(sample['image'], vis_processors).to(device)
        samples.append(sample)

    # run ex
    out_samples = run_model(args, samples, model, call_model_engine, tokenizer, processor)

    save_json(args.output_path, out_samples)
    # metric_dict.update({"num_example": len(out_samples)})
    # save_json(save_result_path, metric_dict)


if __name__ == '__main__':
    main()