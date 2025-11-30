import torch
from transformers import AutoModel, AutoImageProcessor
from transformers.image_utils import load_image
import inspect

from transformers.models.siglip.modeling_siglip import SiglipModel



if __name__ == "__main__":

    ckpt = "google/siglip2-so400m-patch16-256"

    # load model
    model = AutoModel.from_pretrained(ckpt, device_map="cpu").eval()

    # only need the image side for get_image_features
    image_processor = AutoImageProcessor.from_pretrained(ckpt)

    # load the image
    image = load_image(
        "https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg"
    )

    inputs = image_processor(images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)
    print(image_embeddings.shape)
    
    print("Model class:", type(model))
    print("Module:", model.__module__)
    print("Source file:", inspect.getsourcefile(model.__class__))

