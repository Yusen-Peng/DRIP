import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import os
import sys
import numpy as np
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../../../../"))
sys.path.insert(0, PROJECT_ROOT)

from src.open_clip_local.BP import BoundaryPredictor, downsample, H_Net

def complement_idx(idx, dim):
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

outputs = {}
def hook_k(module, input, output):
    outputs['desired_k'] = output

def hook_q(module, input, output):
    outputs['desired_q'] = output



class CLIPVisionTower(nn.Module):
    def __init__(self, 
            vision_tower, 
            args,
            merge_strategy="ViT",
            compression_rate=None, # None or a float number
            drip_weight_path=None,
            temperature=None,
            delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.merge_strategy = merge_strategy
        self.compression_rate = compression_rate
        self.drip_weight_path = drip_weight_path
        self.temperature = temperature
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
    
    def load_drip_weights(self, drip_weight_path):
        print(f"🌊🌊🌊 [INFO] Loading DRIP weights from {drip_weight_path}")
        sd = torch.load(drip_weight_path, map_location="cpu")

        bp_anchor = "vision_tower.boundary_predictor."
        null_suffix = "vision_tower.null_token"

        bp_sd = {}
        null_tensor = None

        for k, v in sd.items():
            if bp_anchor in k:
                # keep only BoundaryPredictor's internal keys:
                # boundary_predictor.0.weight, boundary_predictor.0.bias, ...
                new_k = k.split(bp_anchor, 1)[1]
                bp_sd[new_k] = v

            if k.endswith(null_suffix):
                null_tensor = v

        print("🌊🌊🌊 [INFO] Loaded BP keys:")
        for k in bp_sd.keys():
            print(f"    {k}")

        if len(bp_sd) == 0:
            raise RuntimeError(
                f"No boundary_predictor weights found in {drip_weight_path}. "
                f"First keys: {list(sd.keys())[:10]}"
            )

        missing, unexpected = self.boundary_predictor.load_state_dict(bp_sd, strict=True)

        if null_tensor is not None:
            with torch.no_grad():
                self.null_token.copy_(null_tensor)
            print("🌊🌊🌊 [INFO] Loaded null_token")
        else:
            print("⚠️ [INFO] null_token not found in drip.bin")
        if missing:
            print(f"⚠️ [INFO] Missing BP keys: {missing}")
        if unexpected:
            print(f"⚠️ [INFO] Unexpected BP keys: {unexpected}")
        return missing, unexpected

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        print(f"🍑🍑🍑🍑 [INFO] Loaded image processor for {self.vision_tower_name} with resolution: {self.image_processor.size}")
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

        if self.merge_strategy == "DRIP" or self.merge_strategy == "DRIP-H":
            assert self.compression_rate is not None, "Compression rate must be provided for DRIP merge strategy."
            width = self.vision_tower.config.hidden_size
            mlp_ratio = self.vision_tower.config.intermediate_size / self.vision_tower.config.hidden_size
            self.null_token = nn.Parameter(torch.zeros(1, 1, width))
            
            if self.merge_strategy == "DRIP-H":
                self.boundary_predictor = H_Net(
                    d_model=width,
                    d_inner=int(width * mlp_ratio),
                    activation_function="gelu",
                    temp=self.temperature,
                    prior=self.compression_rate,
                    bp_type='gumbel',
                    threshold=0.5,
                    smart_init=False
                )
                print(f"🐶🐶🐶 [INFO] Using DRIP H-Net merge strategy with compression rate {self.compression_rate}. This will on average keep {max(1, int(1/self.compression_rate))} tokens.")
                print(f"🌪🌪🌪 [INFO] sampling temperature during training: {self.temperature}")
            else:
                self.boundary_predictor = BoundaryPredictor(
                    d_model=width,
                    d_inner=int(width * mlp_ratio),
                    activation_function="gelu",
                    temp=self.temperature,
                    prior=self.compression_rate,
                    bp_type='gumbel',
                    threshold=0.5,
                    smart_init=False
                )
                print(f"🐰🐰🐰 [INFO] Using DRIP merge strategy with compression rate {self.compression_rate}. This will on average keep {max(1, int(1/self.compression_rate))} tokens.")
                print(f"🌪🌪🌪 [INFO] sampling temperature during training: {self.temperature}")

            if self.drip_weight_path is not None:
                missing, unexpected = self.load_drip_weights(self.drip_weight_path)
                assert len(missing) == 0, f"Missing keys when loading DRIP weights: {missing}"
                assert len(unexpected) == 0, f"Unexpected keys when loading DRIP weights: {unexpected}"
                print(f"🦄🦄🦄 [INFO] Loaded DRIP weights from {self.drip_weight_path}")
            else:
                print(f"🐴🐴🐴 [INFO] No DRIP weights provided, initializing DRIP modules from scratch.")            


        elif self.merge_strategy == "Fixed":
            assert self.compression_rate is not None, "compression_rate must be provided for Fixed merge strategy."
            width = self.vision_tower.config.hidden_size
            self.null_token = nn.Parameter(torch.zeros(1, 1, width))
            print(f"🐰🐰🐰 [INFO] Using Fixed merge strategy with compression rate {self.compression_rate}. This will keep every {max(1, int(1/self.compression_rate))} tokens.")
        
        elif self.merge_strategy == "PruMerge":
            assert self.compression_rate is not None, "compression_rate must be provided for PruMerge merge strategy."
            print(f"🐰🐰🐰 [INFO] Using LLaVA-PruMerge strategy with compression rate {self.compression_rate}. This will on average keep {max(1, int(1/self.compression_rate))} tokens")



        else:
            # no additional modules needed for plain ViT
            print(f"🩵🩵🩵 [INFO] Using original ViT features without merging. This will keep all tokens ({self.num_patches} tokens).")

    def _merge_patch_tokens(self, patch_tokens: torch.Tensor, inference=False):
        B, L, D = patch_tokens.shape

        if self.merge_strategy == "Fixed":
            num_tokens_to_keep = max(1, int(L * self.compression_rate))
            indices = torch.linspace(0, L - 1, steps=num_tokens_to_keep, device=patch_tokens.device).round().long()
            hard_boundaries = torch.zeros(B, L, device=patch_tokens.device)
            hard_boundaries[:, indices] = 1

        elif self.merge_strategy  == "DRIP" or self.merge_strategy == "DRIP-H":
            patch_transposed = patch_tokens.transpose(0, 1)  # [L, B, D]

            if hasattr(self, "boundary_predictor"):
                self.boundary_predictor.to(device=patch_tokens.device, dtype=patch_tokens.dtype)

            if hasattr(self, "null_token"):
                self.null_token.data = self.null_token.data.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
            
            if inference:
                _, hard_boundaries = self.boundary_predictor.inference(patch_transposed)
            else:
                _, hard_boundaries = self.boundary_predictor(patch_transposed)
            

            """
                enforce the last token to be a boundary token
            """
            last = torch.ones_like(hard_boundaries[:, -1:])
            hard_boundaries = torch.cat([hard_boundaries[:, :-1], last], dim=1)

        else:
            raise ValueError(f'Unknown merge strategy: {self.merge_strategy}')

        hidden = patch_tokens.transpose(0, 1)              # [L, B, D]

        shortened_hidden = downsample(
            boundaries=hard_boundaries,
            hidden=hidden,
            null_group=self.null_token
        )                                            # [S, B, D]

        merged_tokens = shortened_hidden.transpose(0, 1)  # [B, S, D]

        if not inference:
            if self.merge_strategy == "Fixed":
                boundary_loss = patch_tokens.new_zeros(())
            elif self.merge_strategy == "DRIP":
                boundary_loss = self.boundary_predictor.calc_loss(hard_boundaries)
            elif self.merge_strategy == "DRIP-H":
                boundary_loss = self.boundary_predictor.calc_loss(hard_boundaries)
            else:
                raise ValueError(f'Unknown merge strategy: {self.merge_strategy}')
            avg_boundaries_per_batch = hard_boundaries.sum(dim=1).float().mean().item()
            boundary_ratio = avg_boundaries_per_batch / hard_boundaries.size(1)
            return merged_tokens, boundary_loss, avg_boundaries_per_batch, boundary_ratio
        else:
            return merged_tokens

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        image_features = image_features[:, 1:]
        return image_features
    
    def token_prune_merge_advanced(self, images, reduction_ratio):
        '''
            LLaVA PruMerge
            code adapted from: 
            https://github.com/42Shawn/LLaVA-PruMerge/blob/main/llava/model/multimodal_encoder/clip_encoder.py#L85
        '''
        # token_indix_list = []
        # token_indix_dict = {}

        #set hooks for extracting desired layer's k and q
        hook_handle_k = self.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
        hook_handle_q = self.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)

        #forward pass
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        cls_token_last_layer = image_forward_outs.hidden_states[self.select_layer][:, 0:1]
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        B, N, C = image_features.shape # [B, N, C]

        #extract desired layer's k and q and remove hooks; calculate attention
        desired_layer_k = outputs["desired_k"] # [B, N+1, C]
        desired_layer_q = outputs["desired_q"] # [B, N+1, C]

        hook_handle_k.remove()
        hook_handle_q.remove()

        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        attn = F.softmax(attn, dim=-1) # [B, N+1, N+1]

        cls_attn = attn[:, 0, 1:] # [B, N]

        _, idx = torch.topk(cls_attn, int(N*reduction_ratio), dim=1, largest=True)  # [B, left_tokens] , sorted=True
        index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

        Key_wo_cls = desired_layer_k[:, 1:]  # [B, N-1, C]

        x_others = torch.gather(image_features, dim=1, index=index)  # [B, left_tokens, C]
        x_others_attn = torch.gather(cls_attn, dim=1, index=idx)  
        Key_others = torch.gather(Key_wo_cls, dim=1, index=index)  # [B, left_tokens, C]
        compl = complement_idx(idx, N)  # [B, N-1-left_tokens]
        non_topk = torch.gather(image_features, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]
        non_topk_Key = torch.gather(Key_wo_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]

        Key_others_norm = F.normalize(Key_others, p=2, dim=-1)
        non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)

        # cos_sim = torch.bmm(Key_others_norm, non_topk_Key_norm.transpose(1, 2)) # [B, left_tokens, N-1-left_tokens]

        # _, cluster_indices = torch.topk(cos_sim, k=4, dim=2, largest=True)

        B, left_tokens, C = x_others.size()
        updated_x_others = torch.zeros_like(x_others)

        for b in range(B):
            for i in range(left_tokens):
                key_others_norm = Key_others_norm[b,i,:].unsqueeze(0).unsqueeze(0)

                before_i_Key = Key_others_norm[b, :i, :].unsqueeze(0)  
                after_i_Key = Key_others_norm[b, i+1:, :].unsqueeze(0) 

                before_i_x_others = x_others[b, :i, :].unsqueeze(0)  
                after_i_x_others = x_others[b, i+1:, :].unsqueeze(0)   
                rest_x_others = torch.cat([before_i_x_others, after_i_x_others, non_topk[b,:,:].unsqueeze(0)], dim=1)   
                before_i_x_others_attn = x_others_attn[b, :i].unsqueeze(0)  
                after_i_x_others_attn = x_others_attn[b, i+1:].unsqueeze(0)  
                rest_x_others_attn = torch.cat([before_i_x_others_attn, after_i_x_others_attn, non_topk_attn[b,:].unsqueeze(0)], dim=1)  

                rest_Keys = torch.cat([before_i_Key, after_i_Key, non_topk_Key_norm[b,:,:].unsqueeze(0)], dim=1)
                cos_sim_matrix = torch.bmm(key_others_norm, rest_Keys.transpose(1, 2))

                _, cluster_indices = torch.topk(cos_sim_matrix, k=int(32), dim=2, largest=True)


                cluster_tokens = rest_x_others[:,cluster_indices.squeeze(),:]
                weights = rest_x_others_attn[:,cluster_indices.squeeze()].unsqueeze(-1)

                # update cluster centers
                weighted_avg = torch.sum(cluster_tokens * weights, dim=1) #/ torch.sum(weights)
                updated_center = weighted_avg + x_others[b, i, :]  
                updated_x_others[b, i, :] = updated_center 
            

        extra_one_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
        updated_x_others = torch.cat([updated_x_others, extra_one_token],dim=1)


        # NOTE: fix the type mismatch issue
        image_features = updated_x_others.to(dtype=self.dtype)
        return image_features


    def forward(self, images, inference=False):
        if isinstance(images, list):
            image_features = []
            boundary_losses = []

            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)

                if self.merge_strategy in ["DRIP", "Fixed", "DRIP-H"]:
                    if not inference:
                        image_feature, boundary_loss, _, _ = self._merge_patch_tokens(image_feature, inference=False)
                        boundary_losses.append(boundary_loss)
                    else:
                        image_feature = self._merge_patch_tokens(image_feature, inference=True)

                image_features.append(image_feature)

            if not inference and self.merge_strategy in ["DRIP", "Fixed", "DRIP-H"]:
                boundary_loss = torch.stack(boundary_losses).mean()
                return image_features, boundary_loss

            return image_features

        else:
            if self.merge_strategy == "PruMerge":
                image_features = self.token_prune_merge_advanced(images, reduction_ratio=self.compression_rate)
                # NOTE: we need to hardcode the precision/data type after PruMerge
                image_features = image_features.to(dtype=torch.float16)
            else:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                image_features = self.feature_select(image_forward_outs).to(images.dtype)

                if self.merge_strategy in ["DRIP", "Fixed", "DRIP-H"]:
                    if not inference:
                        image_features, boundary_loss, _, _ = self._merge_patch_tokens(image_features, inference=False)
                        return image_features, boundary_loss
                    else:
                        image_features = self._merge_patch_tokens(image_features, inference=True)
            
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
