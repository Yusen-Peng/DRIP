from LLaVA_wrapper.llava_local.model.builder import load_pretrained_model
from LLaVA_wrapper.llava_local.mm_utils import get_model_name_from_path
from LLaVA_wrapper.llava_local.eval.run_llava import eval_model


def uni_inference(
        prompt: str = "What is happening in this image?",
        image_file: str = "uni_test/football.png",
        model_path: str = "liuhaotian/llava-v1.5-7b"
    ):

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    eval_model(args)
    

if __name__ == "__main__":
    uni_inference()
