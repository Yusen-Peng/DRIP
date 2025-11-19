import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, List
import os
import sys
from collections import OrderedDict
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../../../../"))
sys.path.insert(0, PROJECT_ROOT)
from src.open_clip_local.DTP_ViT import DTPViT, SingleAdaptedFixed, SingleAdaptedSwin
from src.open_clip_local.model import VisionTransformer
from src.boundary_vis import load_dtpx_from_clip_checkpoint, load_dtp_from_clip_checkpoint, load_vit_from_clip_checkpoint, weight_transfer

@torch.no_grad()
def load_fixed_pooling(
    model: nn.Module,
    ckpt_path: str,
    map_location: str = "cpu",
    strict_head: bool = True,   # if False, exclude head.* from BOTH ckpt and model
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loader for SingleAdaptedFixed from a CLIP/DRIP-style checkpoint.

    It will:
      - Look for visual / vision_tower weights and strip prefixes:
          * module.visual., visual., module.vision_tower., vision_tower.
      - Optionally remap short_blocks.* -> post_blocks.* for the post stack.
      - Handle patch_embed.proj.* vs flat patch_embed.* depending on model.
      - Optionally drop head.* if strict_head=False.
      - Enforce strict key/shape matching (except for the optional head case).
    """

    # 0) Read checkpoint
    raw = torch.load(ckpt_path, map_location=map_location)
    if isinstance(raw, dict):
        if "state_dict" in raw:
            sd = raw["state_dict"]
        elif "model" in raw and isinstance(raw["model"], dict):
            sd = raw["model"]
        else:
            sd = raw
    else:
        sd = raw

    if not isinstance(sd, dict):
        raise ValueError("Unsupported checkpoint format: expected a state_dict-like mapping.")

    # Model keys
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())

    # Does model expect patch_embed.proj.* or flat patch_embed.*?
    expects_patch_proj = any(k.startswith("patch_embed.proj.") for k in model_keys)
    expects_patch_flat = any(
        k.startswith("patch_embed.weight") or k.startswith("patch_embed.bias")
        for k in model_keys
    )

    # Helper: strip common prefixes
    def strip_prefixes(k: str) -> Optional[str]:
        # Most CLIP-style checkpoints
        if k.startswith("module.visual."):
            return k[len("module.visual."):]
        if k.startswith("visual."):
            return k[len("visual."):]
        # Sometimes saved as vision_tower
        if k.startswith("module.vision_tower."):
            return k[len("module.vision_tower."):]
        if k.startswith("vision_tower."):
            return k[len("vision_tower."):]
        # Already in local SingleAdaptedFixed format
        return k

    # Remap visual / tower side into SingleAdaptedFixed layout
    def remap(inner: str) -> Optional[str]:
        # ----- patch embed -----
        if inner.startswith("patch_embed.proj."):
            # keep proj layout if model expects it; otherwise flatten
            if expects_patch_proj:
                return inner
            if expects_patch_flat:
                return inner.replace("patch_embed.proj.", "patch_embed.", 1)
            # if neither is expected, just keep and let strict check complain
            return inner

        if inner.startswith("patch_embed.") and not inner.startswith("patch_embed.proj."):
            # This is already flat; keep as-is and let strict checks handle mismatches
            return inner

        # ----- positional embeddings -----
        # SingleAdaptedFixed uses pos_pre and pos_post explicitly.
        # If checkpoint has them, keep them; if it only has pos_emb, we just ignore that.
        if inner.startswith(("pos_pre", "pos_post")):
            return inner
        if inner.startswith("pos_emb"):
            # we don't try to split a single pos_emb into pre/post → ignore
            return None

        # ----- transformer blocks -----
        # pre_blocks.* maps directly
        if inner.startswith("pre_blocks."):
            return inner

        # short_blocks.* in ckpt → post_blocks.* in Fixed
        if inner.startswith("short_blocks."):
            return inner.replace("short_blocks.", "post_blocks.", 1)

        # If checkpoint already uses post_blocks.*, keep it
        if inner.startswith("post_blocks."):
            return inner

        # ----- norm + head + merge -----
        if inner.startswith(("post_ln.", "head.", "merge.")):
            return inner

        # Everything else (boundary_predictor, down_ln, null_token, etc.) → drop
        return None

    mapped: Dict[str, torch.Tensor] = OrderedDict()
    for k, v in sd.items():
        inner = strip_prefixes(k)
        if inner is None:
            continue
        mk = remap(inner)
        if mk is None:
            continue
        if (not strict_head) and mk.startswith("head."):
            # Drop head from source if strict_head=False
            continue
        mapped[mk] = v

    # 1) Block index sanity for pre / post
    def _idxs(prefix: str) -> List[int]:
        out = []
        pref = prefix + "."
        for k in mapped.keys():
            if k.startswith(pref):
                try:
                    out.append(int(k.split(".")[1]))
                except Exception:
                    pass
        return sorted(set(out))

    pre_idx  = _idxs("pre_blocks")
    post_idx = _idxs("post_blocks")

    exp_pre  = list(range(len(getattr(model, "pre_blocks", []))))
    exp_post = list(range(len(getattr(model, "post_blocks", []))))

    if verbose:
        print(f"[fixed] pre_blocks in ckpt:  {pre_idx}  (expected {exp_pre})")
        print(f"[fixed] post_blocks in ckpt: {post_idx} (expected {exp_post})")

    if pre_idx != exp_pre or post_idx != exp_post:
        raise RuntimeError(
            f"[fixed] Block index mismatch.\n"
            f"  ckpt pre_blocks:  {pre_idx} vs expected {exp_pre}\n"
            f"  ckpt post_blocks: {post_idx} vs expected {exp_post}"
        )

    # 2) Strict key/shape validation
    target_keys = set(model_keys)
    if not strict_head:
        target_keys = {k for k in target_keys if not k.startswith("head.")}

    src_keys = set(mapped.keys())

    missing = sorted(target_keys - src_keys)
    unexpected = sorted(src_keys - target_keys)

    shape_mismatch: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    for k in sorted(src_keys & target_keys):
        if tuple(model_sd[k].shape) != tuple(mapped[k].shape):
            shape_mismatch.append((k, tuple(mapped[k].shape), tuple(model_sd[k].shape)))

    if missing or unexpected or shape_mismatch:
        lines = ["[fixed] Checkpoint does not exactly match SingleAdaptedFixed."]
        if missing:
            lines.append(f"  - Missing in ckpt ({len(missing)}): {missing[:12]}{' ...' if len(missing)>12 else ''}")
        if unexpected:
            lines.append(f"  - Unexpected in ckpt ({len(unexpected)}): {unexpected[:12]}{' ...' if len(unexpected)>12 else ''}")
        if shape_mismatch:
            preview = [f"{k}: ckpt{ck} vs model{mk}" for k, ck, mk in shape_mismatch[:12]]
            lines.append(
                f"  - Shape mismatches ({len(shape_mismatch)}): "
                + "; ".join(preview)
                + (" ..." if len(shape_mismatch) > 12 else "")
            )
        raise RuntimeError("\n".join(lines))

    # 3) Actually load
    # Use strict=False because we've already manually enforced key/shape matches
    model.load_state_dict({k: mapped[k] for k in sorted(src_keys)}, strict=False)

    if verbose:
        print(f"[fixed] Loaded {len(src_keys)} tensors into SingleAdaptedFixed.")

    info: Dict[str, Any] = {
        "loaded": sorted(src_keys),
        "pre_blocks_indices_ckpt": pre_idx,
        "post_blocks_indices_ckpt": post_idx,
        "strict_head": strict_head,
    }

    return model, info


def load_finetuned_vision_tower(vt_core, path_or_dir, strict: bool=False, device: str=None, dtype=None):
    """
    vt_core: the VisionTransformer module itself (NOT a wrapper).
    path_or_dir: directory containing 'vision_tower.pt' OR a direct '.pt' file path.
    """
    print(f"vt_core type: {type(vt_core)}")
    # Load weights on CPU, then into module
    state = torch.load(path_or_dir, map_location="cpu")

    missing, unexpected = vt_core.load_state_dict(state, strict=strict)
    if missing or unexpected:
        print(f"[vision_tower] load_state_dict: missing={missing}, unexpected={unexpected}", flush=True)

    vt_core.eval() # no gradients

    # Device / dtype placement
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    vt_core.to(device=device)
    if dtype is not None:
        vt_core.to(dtype=dtype)

    print(f"[vision_tower] Loaded finetuned weights from {path_or_dir}", flush=True)
    return vt_core


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class ViTVisionTower(nn.Module):
    """
    DTP ViT wrapper for CLIP-like vision tower.
    This class is designed to load a DTP ViT model from a CLIP checkpoint and
    provide a forward method that returns image features.
    """
    def __init__(self, 
            checkpoint_path: str,
            vision_tower: str,
            args, 
            image_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            hidden_size: int = 768,
            depth: Tuple = (2, 10, 0),
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            drop_rate: float = 0.1,
            attn_drop_rate: float = 0.1, 
            temp: float = 0.5, 
            compression_rate: float = 0.1,
            threshold: float = 0.5,
            lower_bound: bool = False,
            lambda_val: float = 1.0,
            activation_function: str = 'gelu',
            num_classes: int = 512,
            flop_measure: bool = False,
            delay_load=False,
            finetuning_mode: bool = False
            ):
        super().__init__()

        self.vision_tower_name = vision_tower
        self.checkpoint_path = checkpoint_path
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self._hidden_size = hidden_size
        self.attn_drop_rate = attn_drop_rate
        self.temp = temp
        self.compression_rate = compression_rate
        self.threshold = threshold
        self.lower_bound = lower_bound
        self.lambda_val = lambda_val
        self.activation_function = activation_function
        self.num_classes = num_classes
        self.flop_measure = flop_measure
        self.finetuning_mode = finetuning_mode

        self.is_loaded = False
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f"{self.checkpoint_path} is already loaded. Skipping.")
            print(f"btw, device map is {device_map}")
            return
        
        self.vision_tower: VisionTransformer = VisionTransformer(
                image_size=self.image_size,
                patch_size=self.patch_size,
                width=self._hidden_size,
                layers=12,
                heads=self.num_heads,
                mlp_ratio=self.mlp_ratio
        )

        # FIXME: load the model
        # option 1: load from the original CLIP checkpoint
        load_vit_from_clip_checkpoint(self.vision_tower, self.checkpoint_path)

        # option 2: load from the finetuned checkpoint
        #load_finetuned_vision_tower(self.vision_tower, self.checkpoint_path)


        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vision_tower.to(device)

        # if in finetuning mode, change precision into float16
        if self.finetuning_mode:
            self.vision_tower = self.vision_tower.half()
        
        self.vision_tower.requires_grad_(False)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.image_processor.size = {'shortest_edge': 224}
        self.image_processor.crop_size = {'height': 224, 'width': 224}
        self.is_loaded = True
        self.configurations = {
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'in_chans': self.in_chans,
            'hidden_size': self._hidden_size,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'drop_rate': self.drop_rate,
            'attn_drop_rate': self.attn_drop_rate,
            'temp': self.temp,
            'compression_rate': self.compression_rate,
            'threshold': self.threshold,
            'lower_bound': self.lower_bound,
            'lambda_val': self.lambda_val,
            'activation_function': self.activation_function,
            'num_classes': self.num_classes,
            'flop_measure': self.flop_measure
        }
        
    
    def feature_select(self, image_forward_outs):
        assert image_forward_outs is not None
        raise NotImplementedError("DTPViT does not require feature selection like CLIP. Use the full output.")

    @torch.no_grad() # FIXME: remove this if finetuning
    def forward(self, images):
        """
        images: torch.Tensor of shape [B, C, H, W]
        returns: torch.Tensor of shape [B, N_tokens, hidden_dim]
        """
        images = images.to("cuda", dtype=self.dtype)
        features = self.vision_tower.encode(images)
        features = features.to("cuda", dtype=self.dtype)        
        return features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return torch.device("cuda")

    @property
    def config(self):
        return self.configurations

    @property
    def hidden_size(self):
        return self.vision_tower.width

    @property
    def num_patches_per_side(self):
        return self.vision_tower.image_size // self.vision_tower.patch_size

    @property
    def num_patches(self):
        return self.num_patches_per_side ** 2
    

class DRIPVisionTower(nn.Module):
    """
    DTP ViT wrapper for CLIP-like vision tower.
    This class is designed to load a DTP ViT model from a CLIP checkpoint and
    provide a forward method that returns image features.
    """
    def __init__(self, 
            backbone: str,
            checkpoint_path: str,
            vision_tower: str,
            args, 
            image_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            hidden_size: int = 768,
            depth: Tuple = (2, 10, 0),
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            drop_rate: float = 0.1,
            attn_drop_rate: float = 0.1, 
            temp: float = 0.5, 
            compression_rate: float = 0.1,
            threshold: float = 0.5,
            lower_bound: bool = False,
            lambda_val: float = 1.0,
            activation_function: str = 'gelu',
            num_classes: int = 512,
            flop_measure: bool = False,
            delay_load=False,
            finetuning_mode: bool = False
            ):
        super().__init__()
        self.backbone = backbone
        self.vision_tower_name = vision_tower
        self.checkpoint_path = checkpoint_path
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self._hidden_size = hidden_size
        self.attn_drop_rate = attn_drop_rate
        self.temp = temp
        self.compression_rate = compression_rate
        self.threshold = threshold
        self.lower_bound = lower_bound
        self.lambda_val = lambda_val
        self.activation_function = activation_function
        self.num_classes = num_classes
        self.flop_measure = flop_measure
        self.finetuning_mode = finetuning_mode

        self.is_loaded = False
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f"{self.checkpoint_path} is already loaded. Skipping.")
            print(f"btw, device map is {device_map}")
            return

        self.vision_tower: DTPViT = DTPViT(
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self._hidden_size,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            temp=self.temp,
            compression_rate=self.compression_rate,
            threshold=self.threshold,
            #lower_bound=self.lower_bound,
            #lambda_val=self.lambda_val,
            activation_function=self.activation_function,
            num_classes=self.num_classes,
            flop_measure=self.flop_measure
        ) 

        if self.backbone == 'ViT':
            self.vision_tower, _ = load_dtp_from_clip_checkpoint(self.vision_tower, self.checkpoint_path)
        elif self.backbone == 'XL':
            self.vision_tower, _ = load_dtpx_from_clip_checkpoint(self.vision_tower, self.checkpoint_path)
        elif self.backbone == 'ViT-with-weights':
            weight_transfer(self.vision_tower)
        else:
            raise ValueError(f'Unsupported backbone: {self.backbone}')

        # NOTE: option 2: load from the finetuned checkpoint
        #load_finetuned_vision_tower(self.vision_tower, self.checkpoint_path)


        # if in finetuning mode, change precision into float16
        if self.finetuning_mode:
            self.vision_tower = self.vision_tower.half()
        
        self.vision_tower.requires_grad_(False)
        device = "cuda" if torch.cuda.is_available() else "cpu" 
        self.vision_tower.to(device)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.image_processor.size = {'shortest_edge': 224}
        self.image_processor.crop_size = {'height': 224, 'width': 224}
        self.is_loaded = True
        self.configurations = {
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'in_chans': self.in_chans,
            'hidden_size': self._hidden_size,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'drop_rate': self.drop_rate,
            'attn_drop_rate': self.attn_drop_rate,
            'temp': self.temp,
            'compression_rate': self.compression_rate,
            'threshold': self.threshold,
            'lower_bound': self.lower_bound,
            'lambda_val': self.lambda_val,
            'activation_function': self.activation_function,
            'num_classes': self.num_classes,
            'flop_measure': self.flop_measure
        }
    
    def feature_select(self, image_forward_outs):
        assert image_forward_outs is not None
        raise NotImplementedError("DTPViT does not require feature selection like CLIP. Use the full output.")

    @torch.no_grad() # FIXME: remove this if finetuning
    def forward(self, images):
        """
        images: torch.Tensor of shape [B, C, H, W]
        returns: torch.Tensor of shape [B, N_tokens, hidden_dim]
        """
        # encode images
        images = images.to("cuda", dtype=self.dtype)
        features = self.vision_tower.encode(images, return_loss=False)
        features = features.to("cuda", dtype=self.dtype)
        return features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return torch.device("cuda")

    @property
    def config(self):
        return self.configurations

    @property
    def hidden_size(self):
        return self.vision_tower.embed_dim

    @property
    def num_patches_per_side(self):
        return self.vision_tower.image_size // self.vision_tower.patch_size

    @property
    def num_patches(self):
        return self.num_patches_per_side ** 2
    

class BaselineVisionTower(nn.Module):
    def __init__(self, 
            baseline_type: str,
            checkpoint_path: str,
            vision_tower: str,
            args, 
            image_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            hidden_size: int = 768,
            depth: Tuple = (2, 10, 0),
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            drop_rate: float = 0.1,
            attn_drop_rate: float = 0.1, 
            temp: float = 0.5, 
            compression_rate: float = 0.1,
            threshold: float = 0.5,
            lower_bound: bool = False,
            lambda_val: float = 1.0,
            activation_function: str = 'gelu',
            num_classes: int = 512,
            flop_measure: bool = False,
            delay_load=False,
            finetuning_mode: bool = False
            ):
        super().__init__()
        self.backbone_type = baseline_type
        self.vision_tower_name = vision_tower
        self.checkpoint_path = checkpoint_path
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self._hidden_size = hidden_size
        self.attn_drop_rate = attn_drop_rate
        self.temp = temp
        self.compression_rate = compression_rate
        self.threshold = threshold
        self.lower_bound = lower_bound
        self.lambda_val = lambda_val
        self.activation_function = activation_function
        self.num_classes = num_classes
        self.flop_measure = flop_measure
        self.finetuning_mode = finetuning_mode

        self.is_loaded = False
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f"{self.checkpoint_path} is already loaded. Skipping.")
            print(f"btw, device map is {device_map}")
            return
        
        if self.backbone_type == 'Fixed':
            print("🍔🍔🍔🍔🍔 Using Fixed pooling 🍔🍔🍔🍔🍔")
            self.vision_tower: SingleAdaptedFixed = SingleAdaptedFixed(
                image_size=self.image_size,
                patch_size=self.patch_size,
                in_chans=self.in_chans,
                embed_dim=self._hidden_size,
                depth=self.depth,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                drop_rate=self.drop_rate,
                num_classes=self.num_classes,
                activation_function=self.activation_function,
                flop_measure=self.flop_measure
            )
            self.vision_tower, _ = load_fixed_pooling(self.vision_tower, self.checkpoint_path)

        elif self.backbone_type == 'Swin':
            print("🚑🚑🚑🚑🚑 Using Swin pooling 🚑🚑🚑🚑🚑")
            # self.vision_tower: SingleAdaptedSwin = SingleAdaptedSwin(
            #     image_size=self.image_size,
            #     patch_size=self.patch_size,
            #     in_chans=self.in_chans,
            #     embed_dim=self._hidden_size,
            #     depth=self.depth,
            #     num_heads=self.num_heads,
            #     mlp_ratio=self.mlp_ratio,
            #     drop_rate=self.drop_rate,
            #     num_classes=self.num_classes,
            #     activation_function=self.activation_function,
            #     flop_measure=self.flop_measure
            # )
            # self.vision_tower, _ = load_fixed_pooling(self.vision_tower, self.checkpoint_path)
        else:
            raise NotImplementedError(f"Unsupported baseline type: {self.backbone_type}")

        # if in finetuning mode, change precision into float16
        if self.finetuning_mode:
            self.vision_tower = self.vision_tower.half()
        
        self.vision_tower.requires_grad_(False)
        device = "cuda" if torch.cuda.is_available() else "cpu" 
        self.vision_tower.to(device)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.image_processor.size = {'shortest_edge': 224}
        self.image_processor.crop_size = {'height': 224, 'width': 224}
        self.is_loaded = True
        self.configurations = {
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'in_chans': self.in_chans,
            'hidden_size': self._hidden_size,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'drop_rate': self.drop_rate,
            'attn_drop_rate': self.attn_drop_rate,
            'temp': self.temp,
            'compression_rate': self.compression_rate,
            'threshold': self.threshold,
            'lower_bound': self.lower_bound,
            'lambda_val': self.lambda_val,
            'activation_function': self.activation_function,
            'num_classes': self.num_classes,
            'flop_measure': self.flop_measure
        }
    
    def feature_select(self, image_forward_outs):
        assert image_forward_outs is not None
        raise NotImplementedError("DTPViT does not require feature selection like CLIP. Use the full output.")

    @torch.no_grad() # FIXME: remove this if finetuning
    def forward(self, images):
        """
        images: torch.Tensor of shape [B, C, H, W]
        returns: torch.Tensor of shape [B, N_tokens, hidden_dim]
        """
        # encode images
        images = images.to("cuda", dtype=self.dtype)
        features = self.vision_tower.encode(images, return_loss=False)
        features = features.to("cuda", dtype=self.dtype)
        return features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return torch.device("cuda")

    @property
    def config(self):
        return self.configurations

    @property
    def hidden_size(self):
        return self.vision_tower.embed_dim

    @property
    def num_patches_per_side(self):
        return self.vision_tower.image_size // self.vision_tower.patch_size

    @property
    def num_patches(self):
        return self.num_patches_per_side ** 2

class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
