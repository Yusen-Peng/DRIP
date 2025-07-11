import torch
from PIL import Image
from open_clip_local import get_tokenizer, create_model_and_transforms
from open_clip_local.model import CLIP, CLIPVisionCfg, CLIPTextCfg
from open_clip_local.model import DTPViT
from open_clip_local.model import set_model_preprocess_cfg
from boundary_vis import load_dtpx_from_clip_checkpoint

TEXTS = ["a diagram", "a dog", "a cat", "a person", "a car", "a building", "a tree", "a flower", "a bird", "a fish"]


def unit_test_ViT(image_name, preprocess):
    """Unit test for ViT-based CLIP model inference."""

    model, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', DTP_ViT=False)
    model.eval()
    tokenizer = get_tokenizer('ViT-B-32')

    image = preprocess(Image.open(f"unit_inference_images/{image_name}")).unsqueeze(0)
    text_tokens = tokenizer(TEXTS)

    with torch.no_grad(), torch.autocast("cuda"):
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    pred = text_probs.argmax(dim=-1).item()
    print(f"Predicted text by ViT: {TEXTS[pred]}")


def unit_test_DTP_ViT(image_name, preprocess):
    patch_size = 32
    compression = "2x"  # or "4x", "10x"

    # Define vision config matching DTPViT
    vision_cfg = CLIPVisionCfg(
        width=768,
        layers=12,
        patch_size=patch_size,
        image_size=224,
        head_width=64,
    )

    # Define text config to match ViT-B/32
    text_cfg = CLIPTextCfg(
        context_length=77,
        vocab_size=49408,
        width=512,
        heads=8,
        layers=12,
    )

    # Build model with DTPViT as visual encoder
    model = CLIP(
        embed_dim=512,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        DTP_ViT=True,
        quick_gelu=False,
        cast_dtype=torch.float16,
    )
    model.cuda().eval()

    # Load visual encoder weights
    ckpt_path = f"logs/DTP-ViT-{compression}-{patch_size}/checkpoints/epoch_10.pt"

    model.visual = load_dtpx_from_clip_checkpoint(model.visual, ckpt_path)
    tokenizer = get_tokenizer(f'ViT-B-{patch_size}')

    image = preprocess(Image.open(f"unit_inference_images/{image_name}")).unsqueeze(0).cuda()

    text_tokens = tokenizer(TEXTS).cuda()

    with torch.no_grad(), torch.autocast("cuda"):
        # If using DTPViT, encode_image returns (features, loss, ...) tuple
        image_features, _, _, _ = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    pred = text_probs.argmax(dim=-1).item()
    print(f"Predicted text by DTP-ViT: {TEXTS[pred]}")


if __name__ == "__main__":
    
    _, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', DTP_ViT=False)

    # infer ViT first
    unit_test_ViT("Cat.png", preprocess)
    # then test DTP-ViT
    unit_test_DTP_ViT("Cat.png", preprocess)
