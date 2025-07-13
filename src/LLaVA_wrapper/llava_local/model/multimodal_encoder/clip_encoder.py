import torch
import torch.nn as nn
from typing import Tuple
import os
import sys
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../../../../"))
sys.path.insert(0, PROJECT_ROOT)
from src.open_clip_local.DTP_ViT import DTPViT
from src.boundary_vis import load_dtpx_from_clip_checkpoint_float

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


class DRIPVisionTower(nn.Module):
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
            delay_load=False):
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

        self.is_loaded = False
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()

    def load_model(self):
        if self.is_loaded:
            print(f"{self.checkpoint_path} is already loaded. Skipping.")
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
            lower_bound=self.lower_bound,
            lambda_val=self.lambda_val,
            activation_function=self.activation_function,
            num_classes=self.num_classes,
            flop_measure=self.flop_measure
        ) 
        self.vision_tower = load_dtpx_from_clip_checkpoint_float(self.vision_tower, self.checkpoint_path)
        self.vision_tower.requires_grad_(False)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.image_processor.size = {'shortest_edge': 224}
        self.image_processor.crop_size = {'height': 224, 'width': 224}
        self.is_loaded = True
        self.vision_tower.config = {
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

    @torch.no_grad()
    def forward(self, images):
        """
        images: torch.Tensor of shape [B, C, H, W]
        returns: torch.Tensor of shape [B, N_tokens, hidden_dim]
        """
        image_features = self.vision_tower(
            images.to(device=self.device, dtype=torch.float32),
            return_loss=False # we don't have to log the boundary loss here
        )
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        return self.vision_tower.config

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
