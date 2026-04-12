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

from src.open_clip_local.BP import BoundaryPredictor, downsample


####################################################################
# copy pasted from LLaVA-PruMerge
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


def outlier_dectection(attn):
    attn_np = attn.to(dtype=torch.float32).cpu().numpy().flatten()

    Q1 = np.percentile(attn_np, 25)
    Q3 = np.percentile(attn_np, 75)
    IQR = Q3 - Q1

    # lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_indices = np.where((attn_np > upper_bound))[0]

    ratio = len(outlier_indices) / len(attn_np)
    return ratio
####################################################################

class CLIPVisionTower(nn.Module):
    def __init__(self, 
            vision_tower, 
            args,
            merge_strategy="ViT", # "ViT" or "DRIP" or "Fixed"
            compression_rate=None, # None or a float number
            delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.merge_strategy = merge_strategy
        self.compression_rate = compression_rate
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
        print(f"🍑🍑🍑🍑 [INFO] Loaded image processor for {self.vision_tower_name} with resolution: {self.image_processor.size}")
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

        if self.merge_strategy == "DRIP":
            assert self.compression_rate is not None, "Compression rate must be provided for DRIP merge strategy."
            width = self.vision_tower.config.hidden_size
            mlp_ratio = self.vision_tower.config.intermediate_size / self.vision_tower.config.hidden_size
            self.null_token = nn.Parameter(torch.zeros(1, 1, width))
            self.boundary_predictor = BoundaryPredictor(
                d_model=width,
                d_inner=int(width * mlp_ratio),
                activation_function="gelu",
                temp=0.1,
                prior=self.compression_rate,
                bp_type='gumbel',
                threshold=0.5,
                smart_init=False
            )
        elif self.merge_strategy == "Fixed":
            assert self.compression_rate is not None, "compression_rate must be provided for Fixed merge strategy."
            width = self.vision_tower.config.hidden_size
            self.null_token = nn.Parameter(torch.zeros(1, 1, width))
            print(f"🐰🐰🐰 [INFO] Using Fixed merge strategy with compression rate {self.compression_rate}. This will keep every {max(1, int(1/self.compression_rate))} tokens.")
        else:
            pass # no additional modules needed for plain ViT

    def _merge_patch_tokens(self, patch_tokens: torch.Tensor, inference=False):
        B, L, D = patch_tokens.shape

        if self.merge_strategy == "Fixed":
            num_tokens_to_keep = max(1, int(L * self.compression_rate))
            indices = torch.linspace(0, L - 1, steps=num_tokens_to_keep, device=patch_tokens.device).round().long()
            hard_boundaries = torch.zeros(B, L, device=patch_tokens.device)
            hard_boundaries[:, indices] = 1

        elif self.merge_strategy  == "DRIP":
            patch_transposed = patch_tokens.transpose(0, 1)  # [L, B, D]
            if inference:
                _, hard_boundaries = self.boundary_predictor.inference(patch_transposed)
            else:
                _, hard_boundaries = self.boundary_predictor(patch_transposed)
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
    
    def token_prune_merge_advanced(self, images, if_adaptive=True, reduction_ratio = 1/8):
        '''
            LLaVA PruMerge
            copy pasted from: 
            https://github.com/42Shawn/LLaVA-PruMerge/blob/main/llava/model/multimodal_encoder/clip_encoder.py#L85
        '''
        # token_indix_list = []
        # token_indix_dict = {}

        #set hooks for extracting desired layer's k and q
        hook_handle_k = self.vision_tower.vision_model.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
        hook_handle_q = self.vision_tower.vision_model.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)

        #forward pass
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        cls_token_last_layer =image_forward_outs.hidden_states[self.select_layer][:, 0:1]
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        B, N, C = image_features.shape

        #extract desired layer's k and q and remove hooks; calculate attention
        desired_layer_k = outputs["desired_k"]
        desired_layer_q = outputs["desired_q"]

        hook_handle_k.remove()
        hook_handle_q.remove()

        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        attn = F.softmax(attn, dim=-1)

        cls_attn = attn[:, 0, 1:]  

        if if_adaptive:
            reduction_ratio = outlier_dectection(cls_attn)#*3.5
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
        image_features = updated_x_others
        return image_features










    def forward(self, images, inference=False):
        if isinstance(images, list):
            image_features = []
            boundary_losses = []

            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)

                if self.merge_strategy in ["DRIP", "Fixed"]:
                    if not inference:
                        image_feature, boundary_loss, _, _ = self._merge_patch_tokens(image_feature, inference=False)
                        boundary_losses.append(boundary_loss)
                    else:
                        image_feature = self._merge_patch_tokens(image_feature, inference=True)

                image_features.append(image_feature)

            if not inference and self.merge_strategy in ["DRIP", "Fixed"]:
                boundary_loss = torch.stack(boundary_losses).mean()
                return image_features, boundary_loss

            return image_features

        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

            if self.merge_strategy in ["DRIP", "Fixed"]:
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













##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################


# def load_HF_checkpoint_intoViT(hf_state_dict, local_model: VisionTransformer):
#     new_sd = {}

#     # retrieve embeddings
#     new_sd["conv1.weight"] = hf_state_dict["vision_model.embeddings.patch_embedding.weight"]
#     new_sd["class_embedding"] = hf_state_dict["vision_model.embeddings.class_embedding"]
#     new_sd["positional_embedding"] = hf_state_dict["vision_model.embeddings.position_embedding.weight"]
#     new_sd["ln_pre.weight"] = hf_state_dict["vision_model.pre_layrnorm.weight"]
#     new_sd["ln_pre.bias"] = hf_state_dict["vision_model.pre_layrnorm.bias"]
#     new_sd["ln_post.weight"] = hf_state_dict["vision_model.post_layernorm.weight"]
#     new_sd["ln_post.bias"] = hf_state_dict["vision_model.post_layernorm.bias"]

#     # load transformer blocks
#     num_local_blocks = len(local_model.transformer.resblocks)

#     for i in range(num_local_blocks):
#         hf_prefix = f"vision_model.encoder.layers.{i}"
#         local_prefix = f"transformer.resblocks.{i}"

#         # layer norm 1
#         new_sd[f"{local_prefix}.ln_1.weight"] = hf_state_dict[f"{hf_prefix}.layer_norm1.weight"]
#         new_sd[f"{local_prefix}.ln_1.bias"] = hf_state_dict[f"{hf_prefix}.layer_norm1.bias"]

#         # qkv -> in_proj
#         q_w = hf_state_dict[f"{hf_prefix}.self_attn.q_proj.weight"]
#         k_w = hf_state_dict[f"{hf_prefix}.self_attn.k_proj.weight"]
#         v_w = hf_state_dict[f"{hf_prefix}.self_attn.v_proj.weight"]
#         new_sd[f"{local_prefix}.attn.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
#         q_b = hf_state_dict[f"{hf_prefix}.self_attn.q_proj.bias"]
#         k_b = hf_state_dict[f"{hf_prefix}.self_attn.k_proj.bias"]
#         v_b = hf_state_dict[f"{hf_prefix}.self_attn.v_proj.bias"]
#         new_sd[f"{local_prefix}.attn.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)

#         # attention output projection
#         new_sd[f"{local_prefix}.attn.out_proj.weight"] = hf_state_dict[f"{hf_prefix}.self_attn.out_proj.weight"]
#         new_sd[f"{local_prefix}.attn.out_proj.bias"] = hf_state_dict[f"{hf_prefix}.self_attn.out_proj.bias"]

#         # layer norm 2
#         new_sd[f"{local_prefix}.ln_2.weight"] = hf_state_dict[f"{hf_prefix}.layer_norm2.weight"]
#         new_sd[f"{local_prefix}.ln_2.bias"] = hf_state_dict[f"{hf_prefix}.layer_norm2.bias"]

#         # mlp
#         new_sd[f"{local_prefix}.mlp.c_fc.weight"] = hf_state_dict[f"{hf_prefix}.mlp.fc1.weight"]
#         new_sd[f"{local_prefix}.mlp.c_fc.bias"] = hf_state_dict[f"{hf_prefix}.mlp.fc1.bias"]
#         new_sd[f"{local_prefix}.mlp.c_proj.weight"] = hf_state_dict[f"{hf_prefix}.mlp.fc2.weight"]
#         new_sd[f"{local_prefix}.mlp.c_proj.bias"] = hf_state_dict[f"{hf_prefix}.mlp.fc2.bias"]
    
#     return new_sd

# class ViTVisionTower(nn.Module):
#     """load a pretrained ViT for reproducibility purposes."""
#     def __init__(self, 
#             checkpoint_path: str,
#             vision_tower: str,
#             args, 
#             image_size: int = 224,
#             patch_size: int = 16,
#             in_chans: int = 3,
#             hidden_size: int = 768,
#             depth: int = 12,
#             num_heads: int = 12,
#             mlp_ratio: float = 4.0,
#             drop_rate: float = 0.1,
#             attn_drop_rate: float = 0.1, 
#             activation_function: str = 'gelu',
#             delay_load=False,
#             finetuning_mode: bool = False,
#             **kwargs,
#             ):
#         super().__init__()

#         self.vision_tower_name = vision_tower
#         self.checkpoint_path = checkpoint_path
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.in_chans = in_chans
#         self.depth = depth
#         self.num_heads = num_heads
#         self.mlp_ratio = mlp_ratio
#         self.drop_rate = drop_rate
#         self._hidden_size = hidden_size
#         self.attn_drop_rate = attn_drop_rate
#         self.activation_function = activation_function
#         self.finetuning_mode = finetuning_mode

#         self.is_loaded = False
#         if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
#             self.load_model()

#     def load_model(self, device_map=None):
#         if self.is_loaded:
#             print(f"{self.checkpoint_path} is already loaded. Skipping.")
#             print(f"btw, device map is {device_map}")
#             return
        
#         self.vision_tower: VisionTransformer = VisionTransformer(
#                 image_size=self.image_size,
#                 patch_size=self.patch_size,
#                 width=self._hidden_size,
#                 layers=12,
#                 heads=self.num_heads,
#                 mlp_ratio=self.mlp_ratio
#         )

#         """
#             load the model.
#         """
#         hf_vision = CLIPVisionModel.from_pretrained(self.checkpoint_path)
#         hf_sd = hf_vision.state_dict()
#         mapped_sd = load_HF_checkpoint_intoViT(hf_sd, self.vision_tower)
#         missing, unexpected = self.vision_tower.load_state_dict(mapped_sd, strict=False)
#         if len(missing) > 0:
#             print(f"😨😨😨 Warning: missing keys when loading ViT checkpoint: {missing}")
#         if len(unexpected) > 0:
#             print(f"😨😨😨 Warning: unexpected keys when loading ViT checkpoint: {unexpected}")


#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.vision_tower.to(device)

#         # if in finetuning mode, change precision into float16
#         if self.finetuning_mode:
#             self.vision_tower = self.vision_tower.half()
        
#         self.vision_tower.requires_grad_(False)
#         self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

#         # FIXME
#         self.image_processor.size = {'shortest_edge': 224}
#         self.image_processor.crop_size = {'height': 224, 'width': 224}




#         self.is_loaded = True
#         self.configurations = {
#             'image_size': self.image_size,
#             'patch_size': self.patch_size,
#             'in_chans': self.in_chans,
#             'hidden_size': self._hidden_size,
#             'depth': self.depth,
#             'num_heads': self.num_heads,
#             'mlp_ratio': self.mlp_ratio,
#             'drop_rate': self.drop_rate,
#             'attn_drop_rate': self.attn_drop_rate,
#             'activation_function': self.activation_function
#         }
        
    
#     def feature_select(self, image_forward_outs):
#         assert image_forward_outs is not None
#         raise NotImplementedError("DTPViT does not require feature selection like CLIP. Use the full output.")

#     @torch.no_grad() # FIXME: remove this if finetuning
#     def forward(self, images):
#         """
#         images: torch.Tensor of shape [B, C, H, W]
#         returns: torch.Tensor of shape [B, N_tokens, hidden_dim]
#         """
#         images = images.to("cuda", dtype=self.dtype)
#         features = self.vision_tower.encode(images)
#         features = features.to("cuda", dtype=self.dtype)        
#         return features

#     @property
#     def dummy_feature(self):
#         return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

#     @property
#     def dtype(self):
#         return next(self.vision_tower.parameters()).dtype

#     @property
#     def device(self):
#         return torch.device("cuda")

#     @property
#     def config(self):
#         return self.configurations

#     @property
#     def hidden_size(self):
#         return self.vision_tower.width

#     @property
#     def num_patches_per_side(self):
#         return self.vision_tower.image_size // self.vision_tower.patch_size

#     @property
#     def num_patches(self):
#         return self.num_patches_per_side ** 2
    

# class DRIPVisionTower(nn.Module):
#     """
#     DTP ViT wrapper for CLIP-like vision tower.
#     This class is designed to load a DTP ViT model from a CLIP checkpoint and
#     provide a forward method that returns image features.
#     """
#     def __init__(self, 
#             backbone: str,
#             checkpoint_path: str,
#             vision_tower: str,
#             args, 
#             image_size: int = 224,
#             patch_size: int = 16,
#             in_chans: int = 3,
#             hidden_size: int = 768,
#             depth: Tuple = (2, 10, 0),
#             num_heads: int = 12,
#             mlp_ratio: float = 4.0,
#             drop_rate: float = 0.1,
#             attn_drop_rate: float = 0.1, 
#             temp: float = 0.5, 
#             compression_rate: float = 0.1,
#             threshold: float = 0.5,
#             lower_bound: bool = False,
#             lambda_val: float = 1.0,
#             activation_function: str = 'gelu',
#             num_classes: int = 512,
#             flop_measure: bool = False,
#             delay_load=False,
#             finetuning_mode: bool = False
#             ):
#         super().__init__()
#         self.backbone = backbone
#         self.vision_tower_name = vision_tower
#         self.checkpoint_path = checkpoint_path
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.in_chans = in_chans
#         self.depth = depth
#         self.num_heads = num_heads
#         self.mlp_ratio = mlp_ratio
#         self.drop_rate = drop_rate
#         self._hidden_size = hidden_size
#         self.attn_drop_rate = attn_drop_rate
#         self.temp = temp
#         self.compression_rate = compression_rate
#         self.threshold = threshold
#         self.lower_bound = lower_bound
#         self.lambda_val = lambda_val
#         self.activation_function = activation_function
#         self.num_classes = num_classes
#         self.flop_measure = flop_measure
#         self.finetuning_mode = finetuning_mode

#         self.is_loaded = False
#         if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
#             self.load_model()

#     def load_model(self, device_map=None):
#         if self.is_loaded:
#             print(f"{self.checkpoint_path} is already loaded. Skipping.")
#             print(f"btw, device map is {device_map}")
#             return

#         self.vision_tower: DTPViT = DTPViT(
#             image_size=self.image_size,
#             patch_size=self.patch_size,
#             in_chans=self.in_chans,
#             embed_dim=self._hidden_size,
#             depth=self.depth,
#             num_heads=self.num_heads,
#             mlp_ratio=self.mlp_ratio,
#             drop_rate=self.drop_rate,
#             attn_drop_rate=self.attn_drop_rate,
#             temp=self.temp,
#             compression_rate=self.compression_rate,
#             threshold=self.threshold,
#             #lower_bound=self.lower_bound,
#             #lambda_val=self.lambda_val,
#             activation_function=self.activation_function,
#             num_classes=self.num_classes,
#             flop_measure=self.flop_measure
#         ) 


#         # FIXME: load the model


#         # if in finetuning mode, change precision into float16
#         if self.finetuning_mode:
#             self.vision_tower = self.vision_tower.half()
        
#         self.vision_tower.requires_grad_(False)
#         device = "cuda" if torch.cuda.is_available() else "cpu" 
#         self.vision_tower.to(device)
#         self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
#         self.image_processor.size = {'shortest_edge': 224}
#         self.image_processor.crop_size = {'height': 224, 'width': 224}
#         self.is_loaded = True
#         self.configurations = {
#             'image_size': self.image_size,
#             'patch_size': self.patch_size,
#             'in_chans': self.in_chans,
#             'hidden_size': self._hidden_size,
#             'depth': self.depth,
#             'num_heads': self.num_heads,
#             'mlp_ratio': self.mlp_ratio,
#             'drop_rate': self.drop_rate,
#             'attn_drop_rate': self.attn_drop_rate,
#             'temp': self.temp,
#             'compression_rate': self.compression_rate,
#             'threshold': self.threshold,
#             'lower_bound': self.lower_bound,
#             'lambda_val': self.lambda_val,
#             'activation_function': self.activation_function,
#             'num_classes': self.num_classes,
#             'flop_measure': self.flop_measure
#         }
    
#     def feature_select(self, image_forward_outs):
#         assert image_forward_outs is not None
#         raise NotImplementedError("DTPViT does not require feature selection like CLIP. Use the full output.")

#     @torch.no_grad() # FIXME: remove this if finetuning
#     def forward(self, images):
#         """
#         images: torch.Tensor of shape [B, C, H, W]
#         returns: torch.Tensor of shape [B, N_tokens, hidden_dim]
#         """
#         # encode images
#         images = images.to("cuda", dtype=self.dtype)
#         features = self.vision_tower.encode(images, return_loss=False)
#         features = features.to("cuda", dtype=self.dtype)
#         return features

#     @property
#     def dummy_feature(self):
#         return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

#     @property
#     def dtype(self):
#         return next(self.vision_tower.parameters()).dtype

#     @property
#     def device(self):
#         return torch.device("cuda")

#     @property
#     def config(self):
#         return self.configurations

#     @property
#     def hidden_size(self):
#         return self.vision_tower.embed_dim

#     @property
#     def num_patches_per_side(self):
#         return self.vision_tower.image_size // self.vision_tower.patch_size

#     @property
#     def num_patches(self):
#         return self.num_patches_per_side ** 2
    

# class BaselineVisionTower(nn.Module):
#     def __init__(self, 
#             baseline_type: str,
#             backbone: str,
#             checkpoint_path: str,
#             vision_tower: str,
#             args, 
#             image_size: int = 224,
#             patch_size: int = 16,
#             in_chans: int = 3,
#             hidden_size: int = 768,
#             depth: Tuple = (2, 10, 0),
#             num_heads: int = 12,
#             mlp_ratio: float = 4.0,
#             drop_rate: float = 0.1,
#             attn_drop_rate: float = 0.1, 
#             temp: float = 0.5, 
#             compression_rate: float = 0.1,
#             threshold: float = 0.5,
#             lower_bound: bool = False,
#             lambda_val: float = 1.0,
#             activation_function: str = 'gelu',
#             num_classes: int = 512,
#             flop_measure: bool = False,
#             delay_load=False,
#             finetuning_mode: bool = False
#             ):
#         super().__init__()
#         self.backbone_type = baseline_type
#         self.backbone = backbone
#         self.vision_tower_name = vision_tower
#         self.checkpoint_path = checkpoint_path
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.in_chans = in_chans
#         self.depth = depth
#         self.num_heads = num_heads
#         self.mlp_ratio = mlp_ratio
#         self.drop_rate = drop_rate
#         self._hidden_size = hidden_size
#         self.attn_drop_rate = attn_drop_rate
#         self.temp = temp
#         self.compression_rate = compression_rate
#         self.threshold = threshold
#         self.lower_bound = lower_bound
#         self.lambda_val = lambda_val
#         self.activation_function = activation_function
#         self.num_classes = num_classes
#         self.flop_measure = flop_measure
#         self.finetuning_mode = finetuning_mode

#         self.is_loaded = False
#         if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
#             self.load_model()

#     def load_model(self, device_map=None):
#         if self.is_loaded:
#             print(f"{self.checkpoint_path} is already loaded. Skipping.")
#             print(f"btw, device map is {device_map}")
#             return
        
#         if self.backbone_type == 'Fixed':
#             print("🍔🍔🍔🍔🍔 Using Fixed pooling 🍔🍔🍔🍔🍔")
#             # self.vision_tower: SingleAdaptedFixed = SingleAdaptedFixed(
#             #     image_size=self.image_size,
#             #     patch_size=self.patch_size,
#             #     in_chans=self.in_chans,
#             #     embed_dim=self._hidden_size,
#             #     depth=self.depth,
#             #     num_heads=self.num_heads,
#             #     mlp_ratio=self.mlp_ratio,
#             #     drop_rate=self.drop_rate,
#             #     num_classes=self.num_classes,
#             #     activation_function=self.activation_function,
#             #     flop_measure=self.flop_measure
#             # )

#             # if self.backbone == 'own':
#             #     self.vision_tower, _ = load_fixed_pooling(self.vision_tower, self.checkpoint_path)
#             # elif self.backbone == 'pretrained':
#             #     print("🍌🍌🍌🍌🍌🍌🍌🍌using pretrained weights🍌🍌🍌🍌🍌🍌🍌🍌🍌🍌")
#             #     weight_transfer_baseline(self.vision_tower)
#             # else:
#             #     raise NotImplementedError(f"Unsupported backbone type: {self.backbone}")

#         elif self.backbone_type == 'Swin':
#             print("🚑🚑🚑🚑🚑 Using Swin pooling 🚑🚑🚑🚑🚑")
#             # self.vision_tower: SingleAdaptedSwin = SingleAdaptedSwin(
#             #     image_size=self.image_size,
#             #     patch_size=self.patch_size,
#             #     in_chans=self.in_chans,
#             #     embed_dim=self._hidden_size,
#             #     depth=self.depth,
#             #     num_heads=self.num_heads,
#             #     mlp_ratio=self.mlp_ratio,
#             #     drop_rate=self.drop_rate,
#             #     num_classes=self.num_classes,
#             #     activation_function=self.activation_function,
#             #     flop_measure=self.flop_measure
#             # )
#             # self.vision_tower, _ = load_fixed_pooling(self.vision_tower, self.checkpoint_path)
#         else:
#             raise NotImplementedError(f"Unsupported baseline type: {self.backbone_type}")

#         # if in finetuning mode, change precision into float16
#         if self.finetuning_mode:
#             self.vision_tower = self.vision_tower.half()
        
#         self.vision_tower.requires_grad_(False)
#         device = "cuda" if torch.cuda.is_available() else "cpu" 
#         self.vision_tower.to(device)
#         self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
#         self.image_processor.size = {'shortest_edge': 224}
#         self.image_processor.crop_size = {'height': 224, 'width': 224}
#         self.is_loaded = True
#         self.configurations = {
#             'image_size': self.image_size,
#             'patch_size': self.patch_size,
#             'in_chans': self.in_chans,
#             'hidden_size': self._hidden_size,
#             'depth': self.depth,
#             'num_heads': self.num_heads,
#             'mlp_ratio': self.mlp_ratio,
#             'drop_rate': self.drop_rate,
#             'attn_drop_rate': self.attn_drop_rate,
#             'temp': self.temp,
#             'compression_rate': self.compression_rate,
#             'threshold': self.threshold,
#             'lower_bound': self.lower_bound,
#             'lambda_val': self.lambda_val,
#             'activation_function': self.activation_function,
#             'num_classes': self.num_classes,
#             'flop_measure': self.flop_measure
#         }
    
#     def feature_select(self, image_forward_outs):
#         assert image_forward_outs is not None
#         raise NotImplementedError("DTPViT does not require feature selection like CLIP. Use the full output.")

#     @torch.no_grad() # FIXME: remove this if finetuning
#     def forward(self, images):
#         """
#         images: torch.Tensor of shape [B, C, H, W]
#         returns: torch.Tensor of shape [B, N_tokens, hidden_dim]
#         """
#         # encode images
#         images = images.to("cuda", dtype=self.dtype)
#         features = self.vision_tower.encode(images, return_loss=False)
#         features = features.to("cuda", dtype=self.dtype)
#         return features

#     @property
#     def dummy_feature(self):
#         return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

#     @property
#     def dtype(self):
#         return next(self.vision_tower.parameters()).dtype

#     @property
#     def device(self):
#         return torch.device("cuda")

#     @property
#     def config(self):
#         return self.configurations

#     @property
#     def hidden_size(self):
#         return self.vision_tower.embed_dim

#     @property
#     def num_patches_per_side(self):
#         return self.vision_tower.image_size // self.vision_tower.patch_size

#     @property
#     def num_patches(self):
#         return self.num_patches_per_side ** 2


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
