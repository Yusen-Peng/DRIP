import os
import sys

# Automatically add the project root to sys.path
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.LLaVA_wrapper.llava_local.model.language_model.llava_llama import LlavaLlamaForCausalLM
