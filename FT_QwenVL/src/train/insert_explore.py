import os
import torch
from peft import LoraConfig, get_peft_model
import ast
from transformers import (
    AutoConfig,
    Qwen3VLForConditionalGeneration,
)

def main():
    config = AutoConfig.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", trust_remote_code=True)
    m = Qwen3VLForConditionalGeneration(config)
    print("type(m.model):", type(m.model))
    print("type(m.model.visual):", type(m.model.visual))



if __name__ == "__main__":
    main()