from configparser import Error
import os
import sys
import torch
import torch.nn as nn

from transformers import SiglipVisionModel, SiglipVisionConfig, SiglipImageProcessor

from .perceiver_utils import PerceiverResampler

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../../../../"))
sys.path.insert(0, PROJECT_ROOT)

from src.open_clip_local.BP import BoundaryPredictor, downsample, H_Net


class SiglipVisionTower(nn.Module):
    def __init__(self, 
            vision_tower, 
            args,
            merge_strategy="ViT",
            compression_rate=None, # None or a float number
            drip_weight_path=None,
            perceiver_weight_path=None,
            temperature=None,
            delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.merge_strategy = merge_strategy
        self.compression_rate = compression_rate
        self.drip_weight_path = drip_weight_path
        self.perceiver_weight_path = perceiver_weight_path
        self.temperature = temperature
        self.select_layer = args.mm_vision_select_layer

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)


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

    def load_perceiver_weights(self, perceiver_weight_path):
        print(f"🌀🌀🌀 [INFO] Loading Perceiver weights from {perceiver_weight_path}")
        sd = torch.load(perceiver_weight_path, map_location="cpu")
        perceiver_anchor = "vision_tower.perceiver_resampler."

        perceiver_sd = {}
        for k, v in sd.items():
            if perceiver_anchor in k:
                new_k = k.split(perceiver_anchor, 1)[1]
                perceiver_sd[new_k] = v

        print("🌀🌀🌀 [INFO] Loaded Perceiver keys:")
        for k in perceiver_sd.keys():
            print(f"    {k}")
        if len(perceiver_sd) == 0:
            raise RuntimeError(
                f"No perceiver_resampler weights found in {perceiver_weight_path}. "
                f"First keys: {list(sd.keys())[:10]}"
            )

        missing, unexpected = self.perceiver_resampler.load_state_dict(perceiver_sd, strict=True)

        if missing:
            print(f"⚠️ [INFO] Missing Perceiver keys: {missing}")
        if unexpected:
            print(f"⚠️ [INFO] Unexpected Perceiver keys: {unexpected}")
        return missing, unexpected


    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        activation_function = "gelu"
        # activation_function = "silu" # FIXME: small ablation
        print(f"🍋‍🟩🍋‍🟩🍋‍🟩 [INFO] Using activation function: {activation_function}")

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
                    activation_function=activation_function,
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
                    activation_function=activation_function,
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

        elif self.merge_strategy == "PruneSID":
            assert self.compression_rate is not None, "compression_rate must be provided for PruneSID merge strategy."
            print(f"🐰🐰🐰 [INFO] Using PruneSID strategy with compression rate {self.compression_rate}. This will on average keep {max(1, int(1/self.compression_rate))} tokens")
            print("🟢🟢🟢NOTE: PruneSID is implemented in src/LLaVA_wrapper/llava_local/model/builder.py (`load_pretrained_model` function)")

        elif self.merge_strategy == "Perceiver":
            assert self.compression_rate is not None, "compression_rate must be provided for Perceiver merge strategy."
            
            # compute the number of latents
            width = self.vision_tower.config.hidden_size
            num_latents = max(1, int(self.num_patches * self.compression_rate))
            mlp_ratio = self.vision_tower.config.intermediate_size / self.vision_tower.config.hidden_size
            
            # create the PerceiverResampler module
            self.perceiver_resampler = PerceiverResampler(dim=width, num_latents=num_latents, depth=1, ff_mult=int(mlp_ratio))
            print(
                f"🌀🌀🌀 [INFO] Using Perceiver resampler with compression rate {self.compression_rate}. "
                f"This maps {self.num_patches} tokens -> {num_latents} learned latent tokens.")

            if self.perceiver_weight_path is not None:
                missing, unexpected = self.load_perceiver_weights(self.perceiver_weight_path)
                assert len(missing) == 0, f"Missing keys when loading Perceiver weights: {missing}"
                assert len(unexpected) == 0, f"Unexpected keys when loading Perceiver weights: {unexpected}"
                print(f"🦄🦄🦄 [INFO] Loaded Perceiver weights from {self.perceiver_weight_path}")
            else:
                print("🐴🐴🐴 [INFO] No Perceiver weights provided, initializing Perceiver from scratch.")

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

    def feature_select(self, image_forward_outs, output_attentions=False):
        # return image_forward_outs.hidden_states[:-1], image_forward_outs.attentions
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if output_attentions:
            image_attentions = image_forward_outs.attentions[-1]
            image_attentions = image_attentions.mean(dim=-2)
            return image_features, image_attentions
        return image_features

    def forward(self, images, inference=False, output_attentions=False):
        if type(images) is list:
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
                
                elif self.merge_strategy == "Perceiver":
                    self.perceiver_resampler.to(device=image_features.device, dtype=image_features.dtype)
                    image_features = self.perceiver_resampler(image_features)

                image_features.append(image_feature)
            
            if not inference and self.merge_strategy in ["DRIP", "Fixed", "DRIP-H"]:
                boundary_loss = torch.stack(boundary_losses).mean()
                return image_features, boundary_loss
            return image_features
        
        else:
            if self.merge_strategy == "PruMerge":
                raise NotImplementedError("PruMerge is not implemented in SiglipVisionTower due to the absence of CLS token.")
            elif self.merge_strategy == "PruneSID":
                raise NotImplementedError("PruneSID is not implemented in SiglipVisionTower due to the absence of CLS token.")
            else:
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), 
                                                    output_hidden_states=True, output_attentions=output_attentions)
                image_features = self.feature_select(image_forward_outs, output_attentions=output_attentions)
                if not isinstance(image_features, tuple):
                    image_features = image_features.to(images.dtype)
                else:
                    image_features = (image_features[0].to(images.dtype), image_features[1].to(images.dtype))
                
                if self.merge_strategy in ["DRIP", "Fixed", "DRIP-H"]:
                    if not inference:
                        image_features, boundary_loss, _, _ = self._merge_patch_tokens(image_features, inference=False)
                        return image_features, boundary_loss
                    else:
                        image_features = self._merge_patch_tokens(image_features, inference=True)
                
                elif self.merge_strategy == "Perceiver":
                    self.perceiver_resampler.to(device=image_features.device, dtype=image_features.dtype)
                    image_features = self.perceiver_resampler(image_features)

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

