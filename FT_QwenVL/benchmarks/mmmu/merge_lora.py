import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import AutoProcessor, AutoModelForVision2Seq

from peft import PeftModel
import argparse


base_id = "Qwen/Qwen3-VL-2B-Instruct"
lora_dir = "/fs/scratch/PAS2836/yusenpeng_checkpoint/testing_lora_2B"
out_dir  = "/fs/scratch/PAS2836/yusenpeng_checkpoint/testing_lora_2B_merged"


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description="Merge LoRA into base model")
    parser.add_argument("--base_id", type=str, default=base_id, help="Base model ID or path")
    parser.add_argument("--lora_dir", type=str, default=lora_dir, help="Directory of the LoRA model")
    parser.add_argument("--out_dir", type=str, default=out_dir, help="Output directory for merged model")
    args = parser.parse_args()

    # 1) load base multimodal model
    model = AutoModelForVision2Seq.from_pretrained(
        args.base_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 2) load LoRA on top
    model = PeftModel.from_pretrained(model, args.lora_dir)

    # 3) merge
    model = model.merge_and_unload()

    # 4) save merged
    model.save_pretrained(args.out_dir, safe_serialization=True)

    # 5) save processor from base
    processor = AutoProcessor.from_pretrained(args.base_id, trust_remote_code=True)
    processor.save_pretrained(args.out_dir)

    print("✅ Saved merged model to:", args.out_dir)