import os

from LLaVA_wrapper.llava_local.model.builder import load_pretrained_model
from LLaVA_wrapper.llava_local.model.language_model.llava_llama import LlavaLlamaForCausalLM
from LLaVA_wrapper.llava_local.utils import disable_torch_init
from LLaVA_wrapper.llava_local.mm_utils import get_model_name_from_path


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def format_params(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.2f}K"
    return str(n)


def report_module_params(model: LlavaLlamaForCausalLM):
    base = model.get_model()
    vision_tower = base.get_vision_tower()

    llm_params = count_params(base) - count_params(vision_tower) - count_params(base.mm_projector)
    vit_params = count_params(vision_tower.vision_tower)
    projector_params = count_params(base.mm_projector)

    bp_params = 0
    if hasattr(vision_tower, "boundary_predictor"):
        bp_params = count_params(vision_tower.boundary_predictor)

    null_params = 0
    if hasattr(vision_tower, "null_token"):
        null_params = vision_tower.null_token.numel()

    drip_extra_params = bp_params + null_params

    total_params = llm_params + vit_params + projector_params + drip_extra_params

    rows = [
        ("Vicuna-1.5-7B", llm_params),
        ("ViT-L/14", vit_params),
        ("MLP projector", projector_params),
        ("Boundary predictor", bp_params)
    ]

    print("=" * 80)
    print(f"{'Module':<25} {'#Params':>15} {'% Total':>12}")
    print("=" * 80)

    for name, n in rows:
        pct = 100 * n / total_params
        print(f"{name:<25} {format_params(n):>15} {pct:>11.4f}%")

    print("=" * 80)
    print(f"{'Total':<25} {format_params(total_params):>15} {100.0:>11.4f}%")


if __name__ == "__main__":
    disable_torch_init()
    model_path = os.path.expanduser("/fs/scratch/PAS2836/yusenpeng_checkpoint/LLaVA_7B_DRIP_4x_finetune_train_full")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    report_module_params(model)

