import torch # type: ignore
from PIL import Image


import open_clip_local

def main():

    model, _, preprocess = open_clip_local.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()
    tokenizer = open_clip_local.get_tokenizer('ViT-B-32')

    image = preprocess(Image.open("unit_inference_images/CLIP.png")).unsqueeze(0)
    texts = ["a diagram", "a dog", "a cat", "a person", "a car", "a building", "a tree", "a flower", "a bird", "a fish"]
    text_tokens = tokenizer(texts)

    with torch.no_grad(), torch.autocast("cuda"):
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    pred = text_probs.argmax(dim=-1).item()
    print(f"Predicted text: {texts[pred]}")

if __name__ == "__main__":
    main() 
