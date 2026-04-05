import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple
from .transformer import LayerNorm, PatchDropout, AttentionalPooler, Transformer, _expand_token
from .transformer import PositionalEmbedding, RelPartialLearnableDecoderLayer
from .transformer import TransformerXL
from .pos_embed import get_2d_sincos_pos_embed
from .BP import BoundaryPredictor, downsample, RoutingModule
from .utils import to_2tuple


class DTPViT(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            depth: tuple[int],
            compression_rate: float,
            heads: int,
            mlp_ratio: float,
            temp: float,
            flop_measure: bool = False,
            threshold: float = 0.5,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'sin_cos_2d', # 'learnable' or 'sin_cos_2d'
            pool_type: str = 'tok',
            smart_init: bool = False,
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim
        self.width = width
        self.layers = layers
        self.depth = depth
        self.prior = compression_rate
        self.threshold = threshold
        self.temp = temp
        self.flop_measure = flop_measure
        self.null_token = nn.Parameter(torch.zeros(1, 1, width))
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1],\
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError
        self.pos_embed_type = pos_embed_type

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()
        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.down_ln = norm_layer(width)
        self.boundary_predictor = BoundaryPredictor(
            d_model=width,
            d_inner=int(width * mlp_ratio),
            activation_function="gelu",
            temp=temp,
            prior=compression_rate,
            bp_type='gumbel',
            threshold=threshold,
            smart_init=smart_init
        )

        self.transformer_pre = Transformer(
            width,
            self.depth[0],
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        self.transformer_post = Transformer(
            width,
            self.depth[1],
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.transformer_pre.grad_checkpointing = enable
        self.transformer_post.grad_checkpointing = enable
    
    @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {'positional_embedding', 'class_embedding'}
        return no_wd

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def _embeds(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # shape = [*, dim, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # patch dropout (if active)
        x = self.patch_dropout(x)

        # apply norm before transformer
        x = self.ln_pre(x)
        return x

    def _pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        return pooled, tokens
    
    def encode(self, x: torch.Tensor, return_loss: bool, inference: bool = False):
        x = self._embeds(x) # [B, 3, H, W] -> [B, L+1, D]
        x = self.transformer_pre(x, attn_mask=None) # [B, L, D] -> [B, L+1, D]

        # Split CLS and patch tokens
        cls_token = x[:, :1, :]      # [B, 1, D]
        patch_tokens = x[:, 1:, :]   # [B, L, D]

        if self.flop_measure:
            B, L, _ = patch_tokens.shape
            num_tokens_to_keep = max(1, int(L * self.prior))
            indices = torch.linspace(0, L - 1, steps=num_tokens_to_keep).round().long()
            hard_boundaries = torch.zeros(B, L, device=x.device)
            # hard boundaries: [B, L]
            hard_boundaries[:, indices] = 1 
        else:
            patch_transposed = patch_tokens.transpose(0, 1) # [B, L, D] -> [L, B, D]
            if not inference:
                # hard boundaries: [B, L] - sampling during training
                _, hard_boundaries = self.boundary_predictor(patch_transposed) # input is [L, B, D]
            else:
                # during inference, apply thresholding to get hard boundaries
                _, hard_boundaries = self.boundary_predictor.inference(patch_transposed) # input is [L, B, D]


        hidden: torch.Tensor = self.down_ln(patch_tokens) # [B, L, D] -> [B, L, D]
        hidden = hidden.transpose(0, 1) # [B, L, D] -> [L, B, D]
        shortened_hidden = downsample(
            boundaries=hard_boundaries,
            hidden=hidden,
            null_group=self.null_token
        ) # [L, B, D] -> [S, B, D]
        shortened_patches = shortened_hidden.transpose(0, 1) # [S, B, D] -> [B, S, D]

        # Re-attach CLS
        shortened_hidden = torch.cat([cls_token, shortened_patches], dim=1)  # [B, 1+S, D]

        features = self.transformer_post(shortened_hidden, attn_mask=None) # [B, 1+S, D] -> [B, 1+S, D]
        
        if return_loss and not self.flop_measure:
            boundary_loss = self.boundary_predictor.calc_loss(hard_boundaries)
            avg_boundaries_per_batch = hard_boundaries.sum(dim=1).float().mean().item()
            boundary_ratio = avg_boundaries_per_batch / hard_boundaries.size(1)
            return features, boundary_loss, avg_boundaries_per_batch, boundary_ratio
        else:
            return features # [B, S, D]

    def forward(self, x: torch.Tensor, return_loss: bool = False, inference: bool = False):
        features_out = self.encode(x, return_loss=return_loss, inference=inference) # [B, 3, H, W] -> [B, S, D]

        if return_loss and not self.flop_measure:
            # encode returns tuple (features, loss, avg_boundaries, boundary_ratio)
            tensor, boundary_loss, avg_boundaries_per_batch, boundary_ratio = features_out
        else:
            tensor = features_out
        
        pooled, tokens = self._pool(tensor) # [B, S, D] -> [B, D], [B, S, D]
        pooled = pooled @ self.proj # [B, D] -> [B, output_dim]

        if self.output_tokens:
            return pooled, tokens

        if return_loss and not self.flop_measure:
            return pooled, boundary_loss, avg_boundaries_per_batch, boundary_ratio # [B, output_dim]
        else:
            return pooled # [B, output_dim]


####################### fixed pooling baseline #######################
class DTPViT_Fixed(DTPViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, x: torch.Tensor, return_loss: bool, inference: bool = False):
        x = self._embeds(x) # [B, 3, H, W] -> [B, L+1, D]
        x = self.transformer_pre(x, attn_mask=None) # [B, L, D] -> [B, L+1, D]

        # Split CLS and patch tokens
        cls_token = x[:, :1, :]      # [B, 1, D]
        patch_tokens = x[:, 1:, :]   # [B, L, D]
        
        # fixed pooling
        B, L, _ = patch_tokens.shape
        num_tokens_to_keep = max(1, int(L * self.prior))
        indices = torch.linspace(0, L - 1, steps=num_tokens_to_keep).round().long()
        hard_boundaries = torch.zeros(B, L, device=x.device)
        # hard boundaries: [B, L]
        hard_boundaries[:, indices] = 1

        hidden: torch.Tensor = self.down_ln(patch_tokens) # [B, L, D] -> [B, L, D]
        hidden = hidden.transpose(0, 1) # [B, L, D] -> [L, B, D]
        shortened_hidden = downsample(
            boundaries=hard_boundaries,
            hidden=hidden,
            null_group=self.null_token
        ) # [L, B, D] -> [S, B, D]
        shortened_patches = shortened_hidden.transpose(0, 1) # [S, B, D] -> [B, S, D]

        # Re-attach CLS
        shortened_hidden = torch.cat([cls_token, shortened_patches], dim=1)  # [B, 1+S, D]

        features = self.transformer_post(shortened_hidden, attn_mask=None) # [B, 1+S, D] -> [B, 1+S, D]
        
        if return_loss and not self.flop_measure:
            boundary_loss = self.boundary_predictor.calc_loss(hard_boundaries)
            avg_boundaries_per_batch = hard_boundaries.sum(dim=1).float().mean().item()
            boundary_ratio = avg_boundaries_per_batch / hard_boundaries.size(1)
            return features, boundary_loss, avg_boundaries_per_batch, boundary_ratio
        else:
            return features # [B, S, D]

####################### causal ViT DRIP #######################

class DTPViT_Causal(DTPViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # additive mask for attention: 0 on and below diagonal, -inf above diagonal
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def encode(self, x: torch.Tensor, return_loss: bool):
        x = self._embeds(x)  # [B, L, D]

        pre_mask = self._build_causal_mask(
            seq_len=x.size(1),
            device=x.device,
            dtype=x.dtype,
        )

        x = self.transformer_pre(x, attn_mask=pre_mask)  # [B, L, D]

        if self.flop_measure:
            B, L, _ = x.shape
            num_tokens_to_keep = max(1, int(L * self.prior))
            indices = torch.linspace(0, L - 1, steps=num_tokens_to_keep, device=x.device).round().long()
            hard_boundaries = torch.zeros(B, L, device=x.device)
            hard_boundaries[:, indices] = 1
        else:
            x_transposed = x.transpose(0, 1)  # [L, B, D]
            _, hard_boundaries = self.boundary_predictor(x_transposed)  # [B, L]

        hidden = self.down_ln(x)              # [B, L, D]
        hidden = hidden.transpose(0, 1)       # [L, B, D]

        shortened_hidden = downsample(
            boundaries=hard_boundaries,
            hidden=hidden,
            null_group=self.null_token
        )                                     # [S, B, D]

        shortened_hidden = shortened_hidden.transpose(0, 1)  # [B, S, D]
        
        post_mask = self._build_causal_mask(
            seq_len=shortened_hidden.size(1),
            device=shortened_hidden.device,
            dtype=shortened_hidden.dtype,
        )
        
        features = self.transformer_post(shortened_hidden, attn_mask=post_mask)
        if return_loss and not self.flop_measure:
            boundary_loss = self.boundary_predictor.calc_loss(hard_boundaries)
            avg_boundaries_per_batch = hard_boundaries.sum(dim=1).float().mean().item()
            boundary_ratio = avg_boundaries_per_batch / hard_boundaries.size(1)
            return features, boundary_loss, avg_boundaries_per_batch, boundary_ratio
        else:
            return features


####################### H-Net routing module style DRIP #######################

class DTPViT_CosSim(DTPViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.boundary_predictor = RoutingModule(
            prior=self.prior,
            d_model=self.width,
        )
    

    def encode(self, x: torch.Tensor, return_loss: bool):
        x = self._embeds(x) # [B, 3, H, W] -> [B, L, D]
        x = self.transformer_pre(x, attn_mask=None) # [B, L, D] -> [B, L, D]
        if self.flop_measure:
            B, L, _ = x.shape
            num_tokens_to_keep = max(1, int(L * self.prior))
            indices = torch.linspace(0, L - 1, steps=num_tokens_to_keep).round().long()
            hard_boundaries = torch.zeros(B, L, device=x.device)
            # hard boundaries: [B, L]
            hard_boundaries[:, indices] = 1 
        else:
            x_transposed = x.transpose(0, 1) # [B, L, D] -> [L, B, D]
            # hard boundaries: [B, L]
            soft_boundaries, hard_boundaries = self.boundary_predictor(x_transposed) # input is [L, B, D]
            # print("soft boundaries", soft_boundaries, flush=True)
            # print("hard boundaries", hard_boundaries, flush=True)

        hidden: torch.Tensor = self.down_ln(x) # [B, L, D] -> [B, L, D]
        hidden = hidden.transpose(0, 1) # [B, L, D] -> [L, B, D]
        shortened_hidden = downsample(
            boundaries=hard_boundaries,
            hidden=hidden,
            null_group=self.null_token
        ) # [L, B, D] -> [S, B, D]
        shortened_hidden = shortened_hidden.transpose(0, 1) # [S, B, D] -> [B, S, D]

        features = self.transformer_post(shortened_hidden, attn_mask=None) # [B, S, D] -> [B, S, D]
        
        if return_loss and not self.flop_measure:
            boundary_loss = self.boundary_predictor.calc_loss(soft_boundaries, hard_boundaries)
            avg_boundaries_per_batch = hard_boundaries.sum(dim=1).float().mean().item()
            boundary_ratio = avg_boundaries_per_batch / hard_boundaries.size(1)
            return features, boundary_loss, avg_boundaries_per_batch, boundary_ratio
        else:
            return features # [B, S, D]











###############################################################################################
##


class SingleAdaptedFixed(nn.Module):
    def __init__(self,
                image_size: int,
                patch_size: int,
                width: int,
                layers: int,
                depth: tuple[int],
                compression_rate: float,
                heads: int,
                mlp_ratio: float,
                temp: float,
                flop_measure: bool = False,
                threshold: float = 0.5,
                ls_init_value: float = None,
                attentional_pool: bool = False,
                attn_pooler_queries: int = 256,
                attn_pooler_heads: int = 8,
                output_dim: int = 512,
                patch_dropout: float = 0.1,
                no_ln_pre: bool = False,
                pos_embed_type: str = 'transformer-xl', # 'learnable' or 'sin_cos_2d' or 'transformer-xl'
                pool_type: str = 'avg',
                final_ln_after_pool: bool = False,
                act_layer: Callable = nn.GELU,
                norm_layer: Callable = LayerNorm,
                output_tokens: bool = False,
    ):

        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim
        self.width = width
        self.layers = layers
        self.depth = depth
        self.prior = compression_rate
        self.threshold = threshold
        self.temp = temp
        self.flop_measure = flop_measure
        self.null_token = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.null_token, std=0.02)


        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            # bias=False # NOTE: False
        )

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1],\
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        elif pos_embed_type == 'transformer-xl':
            self.positional_embedding = PositionalEmbedding(demb=width)
        else:
            raise ValueError
        self.pos_embed_type = pos_embed_type
    
        # pos-aware attention bias terms
        num_heads = heads
        embed_dim = width
        self.r_w_bias = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))
        self.r_r_bias = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = nn.Identity()
        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.down_ln = norm_layer(width)

        self.transformer_pre = TransformerXL(
            width,
            self.depth[0],
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            batch_first=False
        )

        self.transformer_post = TransformerXL(
            width,
            self.depth[1],
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            batch_first=False
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))
        self.head = nn.Linear(embed_dim, 1000)


    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.transformer_pre.grad_checkpointing = enable
        self.transformer_post.grad_checkpointing = enable
    
    @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {'positional_embedding', 'class_embedding'}
        return no_wd

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def _embeds(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # shape = [*, dim, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)

        # patch dropout (if active)
        x = self.patch_dropout(x)

        # apply norm before transformer
        x = self.ln_pre(x)
        return x

    def _pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        return pooled, tokens
    
    def encode(self, x: torch.Tensor, return_loss: bool):
        x = self._embeds(x) # [B, 3, H, W] -> [B, L, D]

        # Compute position embeddings
        B, T = x.size(0), x.size(1)
        pos_seq = torch.arange(T - 1, -1, -1.0, device=x.device, dtype=x.dtype)
        pos_emb = self.positional_embedding(pos_seq)
        x = self.transformer_pre(
            x, 
            pos_emb=pos_emb, 
            r_w_bias=self.r_w_bias, 
            r_r_bias=self.r_r_bias) # [B, L, D] -> [B, L, D]
    

        # fixed pooling
        num_tokens_to_keep = max(1, int(T * self.prior))
        indices = torch.linspace(0, T - 1, steps=num_tokens_to_keep).round().long()
        hard_boundaries = torch.zeros(B, T, device=x.device)
        # hard boundaries: [B, T]
        hard_boundaries[:, indices] = 1

        hidden: torch.Tensor = self.down_ln(x) # [B, L, D] -> [B, L, D]
        hidden = hidden.transpose(0, 1) # [B, L, D] -> [L, B, D]
        shortened_hidden = downsample(
            boundaries=hard_boundaries,
            hidden=hidden,
            null_group=self.null_token
        ) # [L, B, D] -> [S, B, D]
        shortened_hidden = shortened_hidden.transpose(0, 1) # [S, B, D] -> [B, S, D]

        S = shortened_hidden.size(1)
        new_pos_seq = torch.arange(S - 1, -1, -1.0, device=x.device, dtype=x.dtype)
        new_pos_emb = self.positional_embedding(new_pos_seq)

        features = self.transformer_post(
            shortened_hidden,
            pos_emb=new_pos_emb, 
            r_w_bias=self.r_w_bias, 
            r_r_bias=self.r_r_bias) # [B, S, D] -> [B, S, D]
        
        if return_loss and not self.flop_measure:
            boundary_loss = torch.tensor(0.0, device=x.device) # entropy-based method: boundary loss disabled
            # boundary_loss = self.boundary_predictor.calc_loss(hard_boundaries)
            avg_boundaries_per_batch = hard_boundaries.sum(dim=1).float().mean().item()
            boundary_ratio = avg_boundaries_per_batch / hard_boundaries.size(1)
            return features, boundary_loss, avg_boundaries_per_batch, boundary_ratio
        else:
            return features # [B, S, D]

    def forward(self, x: torch.Tensor, return_loss: bool = False):
        features_out = self.encode(x, return_loss=return_loss) # [B, 3, H, W] -> [B, S, D]

        if return_loss and not self.flop_measure:
            # encode returns tuple (features, loss, avg_boundaries, boundary_ratio)
            tensor, boundary_loss, avg_boundaries_per_batch, boundary_ratio = features_out
        else:
            tensor = features_out
        
        pooled, tokens = self._pool(tensor) # [B, S, D] -> [B, D], [B, S, D]
        pooled = pooled @ self.proj # [B, D] -> [B, output_dim]

        if self.output_tokens:
            return pooled, tokens

        if return_loss and not self.flop_measure:
            return pooled, boundary_loss, avg_boundaries_per_batch, boundary_ratio # [B, output_dim]
        else:
            return pooled # [B, output_dim]


class XL_Baseline(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            depth: int,
            compression_rate: float,
            heads: int,
            mlp_ratio: float,
            temp: float,
            flop_measure: bool = False,
            threshold: float = 0.5,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.1,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'transformer-xl', # 'learnable' or 'sin_cos_2d' or 'transformer-xl'
            pool_type: str = 'avg',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim
        self.width = width
        self.layers = layers
        self.depth = depth
        self.prior = compression_rate
        self.threshold = threshold
        self.temp = temp
        self.flop_measure = flop_measure
        self.null_token = nn.Parameter(torch.zeros(1, 1, width))
        # NOTE: may be different
        nn.init.normal_(self.null_token, std=0.02)

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            # bias=False # NOTE: False
        )

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1],\
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        elif pos_embed_type == 'transformer-xl':
            self.positional_embedding = PositionalEmbedding(demb=width)
        else:
            raise ValueError
        self.pos_embed_type = pos_embed_type
    
        # pos-aware attention bias terms
        num_heads = heads
        embed_dim = width
        self.r_w_bias = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))
        self.r_r_bias = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()
        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.down_ln = norm_layer(width)

        self.transformer = TransformerXL(
            width,
            self.depth,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            batch_first=False
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))
        self.head = nn.Linear(embed_dim, output_dim)


    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.transformer.grad_checkpointing = enable
    
    @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {'positional_embedding', 'class_embedding'}
        return no_wd

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def _embeds(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # shape = [*, dim, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)

        # patch dropout (if active)
        x = self.patch_dropout(x)

        # apply norm before transformer
        x = self.ln_pre(x)
        return x

    def _pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        return pooled, tokens
    
    def encode(self, x: torch.Tensor):
        x = self._embeds(x) # [B, 3, H, W] -> [B, L, D]

        # Compute position embeddings
        T = x.size(1)
        pos_seq = torch.arange(T - 1, -1, -1.0, device=x.device, dtype=x.dtype)
        pos_emb = self.positional_embedding(pos_seq)
        features = self.transformer(
            x, 
            pos_emb=pos_emb, 
            r_w_bias=self.r_w_bias, 
            r_r_bias=self.r_r_bias) # [B, L, D] -> [B, L, D]

        return features # [B, S, D]

    def forward(self, x: torch.Tensor):
        features_out = self.encode(x) # [B, 3, H, W] -> [B, S, D]

        tensor = features_out
        
        pooled, tokens = self._pool(tensor) # [B, S, D] -> [B, D], [B, S, D]
        pooled = pooled @ self.proj # [B, D] -> [B, output_dim]

        if self.output_tokens:
            return pooled, tokens
        return pooled # [B, output_dim]



########################################################################################################
########################################################################################################
# legacy code below --- IGNORE ---
########################################################################################################


class HierarchicalDTPViT(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=768,
                 depth=(2, 2, 6, 2),
                 num_heads=[3, 6, 12, 24],
                 mlp_ratio=4.0,
                 drop_rate=0.1,
                 attn_drop_rate=0.1,
                 temp=1.0,
                 compression_rate=(0.25, 0.25, 0.25),
                 bp_type='gumbel',
                 threshold=0.5,
                 num_classes=1000,
                 activation_function='gelu',
                 flop_measure: bool = False,
        ):
        super().__init__()
        self.flop_measure = flop_measure
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.seq_len = self.num_patches

        # Patch embedding
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_chans, embed_dim)
        self.dropout = nn.Dropout(drop_rate)

        # Positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))

        def create_decoder_layers(n_layers, n_head):
            layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=n_head,
                        dim_feedforward=int(embed_dim * mlp_ratio),
                        dropout=drop_rate,
                        activation=activation_function,
                        batch_first=False,
                        norm_first=True
                    )
                    for _ in range(n_layers)
                ]
            )

            return layers

        # Transformer blocks for each stage
        self.pre_blocks = create_decoder_layers(depth[0], n_head=num_heads[0])
        self.mid_blocks_1 = create_decoder_layers(depth[1], n_head=num_heads[1])
        self.mid_blocks_2 = create_decoder_layers(depth[2], n_head=num_heads[2])
        self.final_blocks = create_decoder_layers(depth[3], n_head=num_heads[3])

        # Two-stage boundary predictors
        self.bp1 = BoundaryPredictor(
            d_model=embed_dim,
            d_inner=int(embed_dim * mlp_ratio),
            activation_function=activation_function,
            temp=temp,
            prior=compression_rate[0],
            bp_type=bp_type,
            threshold=threshold
        )

        self.bp2 = BoundaryPredictor(
            d_model=embed_dim,
            d_inner=int(embed_dim * mlp_ratio),
            activation_function=activation_function,
            temp=temp,
            prior=compression_rate[1],
            bp_type=bp_type,
            threshold=threshold
        )

        self.bp3 = BoundaryPredictor(
            d_model=embed_dim,
            d_inner=int(embed_dim * mlp_ratio),
            activation_function=activation_function,
            temp=temp,
            prior=compression_rate[2],
            bp_type=bp_type,
            threshold=threshold
        )

        # Layer norm
        self.down_ln = nn.LayerNorm(embed_dim)
        self.null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.null_token, std=0.02)

        # Final classification head
        self.head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes

    def forward_after_pooling_with_attn_masks(self, core_input: torch.Tensor, layers, attention_mask: torch.Tensor):
        """
        Process input with relative attention and padding-aware masking.
        """
        T, B, D = core_input.size()
        core_out = core_input
        for layer in layers:
            core_out = layer(core_out, src_key_padding_mask=attention_mask)
        return core_out
    

    def _make_attn_mask(self, shortened_hidden: torch.Tensor):
        """
        seq: S x B x D
        return: B x S x S mask (True=mask)
        """
        S = shortened_hidden.size(0)
        pad_mask = shortened_hidden.abs().sum(-1).eq(0)       # S x B (1 where padded, 0 where regular)
        attn_mask = pad_mask.transpose(0, 1)                  # (B, S)  True=PAD
        return attn_mask

    def _downsample_stage(self, x: torch.Tensor, boundary_predictor: BoundaryPredictor):
        """
        One stage of boundary prediction + downsampling
        """
        B = x.size(1)
        L = x.size(0)
        hidden = self.down_ln(x)

        # boundary prediction
        if self.flop_measure:
            num_tokens_to_keep = max(1, int(L * boundary_predictor.prior))
            indices = torch.arange(0, L, step=max(1, L // num_tokens_to_keep), device=x.device)
            hard_boundaries = torch.zeros(B, L, device=x.device)
            hard_boundaries[:, indices] = 1
        else:
            _, hard_boundaries = boundary_predictor(x)  # B x L

        # downsample
        shortened_hidden = downsample(
            boundaries=hard_boundaries,
            hidden=hidden,
            null_group=self.null_token
        )  # S x B x D

        return shortened_hidden, hard_boundaries

    def encode(self, x: torch.Tensor, return_loss: bool = False):
        B = x.size(0)

        # ✅ Patch embedding already gives (B, L, C)
        x = self.patch_embed(x)                  # (B, L, C)
        x = self.dropout(x)                      # (B, L, C)

        # Positional embedding (for pre-blocks)
        L = x.size(1)
        pos = self.pos_emb[:, 1:1 + L, :].to(device=x.device, dtype=x.dtype)   # (1, L, C)
        x = x + pos                                                             # (B, L, C)
        # Pre-pooling transformer blocks
        x = x.transpose(0, 1)                    # (L, B, C)
        for block in self.pre_blocks:
            x = block(x)

        x, hard_boundaries1 = self._downsample_stage(x, self.bp1)
        attn_mask1 = self._make_attn_mask(x)

        x = self.forward_after_pooling_with_attn_masks(x, self.mid_blocks_1, attention_mask=attn_mask1)

        x, hard_boundaries2 = self._downsample_stage(x, self.bp2)
        attn_mask2 = self._make_attn_mask(x)

        x = self.forward_after_pooling_with_attn_masks(x, self.mid_blocks_2, attention_mask=attn_mask2)

        x, hard_boundaries3 = self._downsample_stage(x, self.bp3)
        attn_mask3 = self._make_attn_mask(x)

        x = self.forward_after_pooling_with_attn_masks(x, self.final_blocks, attention_mask=attn_mask3)
        features = x  # S x B x D

        if return_loss and not self.flop_measure:
            loss1 = self.bp1.calc_loss(hard_boundaries1)
            loss2 = self.bp2.calc_loss(hard_boundaries2)
            loss3 = self.bp3.calc_loss(hard_boundaries3)
            boundary_loss = loss1 + loss2 + loss3
            avg_boundaries_per_batch1 = hard_boundaries1.sum(dim=1).float().mean().item()
            avg_boundaries_per_batch2 = hard_boundaries2.sum(dim=1).float().mean().item()
            avg_boundaries_per_batch3 = hard_boundaries3.sum(dim=1).float().mean().item()

            boundary_ratio1 = avg_boundaries_per_batch1 / hard_boundaries1.size(1)
            boundary_ratio2 = avg_boundaries_per_batch2 / hard_boundaries2.size(1)
            boundary_ratio3 = avg_boundaries_per_batch3 / hard_boundaries3.size(1)

            # only report the second boundary ratio
            # this is not really that helpful, but we keep it for consistency
            cumulative_avg_boundaries_per_batch = avg_boundaries_per_batch2

            # compute the cumulative boundary ratio (e.g., 0.5 * 0.5 = 0.25)
            # NOTE: this is really important to monitor the cumulative compression ratio!
            cumulative_boundary_ratio = boundary_ratio1 * boundary_ratio2 * boundary_ratio3

            return features, boundary_loss, cumulative_avg_boundaries_per_batch, cumulative_boundary_ratio
        else:
            return features

    def forward(self, x, return_loss=False):
        """
        Full forward pass including pooling to class logits.
        """
        features_out = self.encode(x, return_loss=return_loss)

        if return_loss and not self.flop_measure:
            # encode returns tuple (features, loss, avg_boundaries, boundary_ratio)
            x, boundary_loss, avg_boundaries_per_batch, boundary_ratio = features_out
        else:
            x = features_out

        # pool across sequence dimension with mean pooling
        pad_mask = x.abs().sum(-1).eq(0).float()           # S x B
        valid_mask = 1.0 - pad_mask                        # S x B
        valid_mask_exp = valid_mask.unsqueeze(-1)          # S x B x 1

        x = x * valid_mask_exp                             # Mask padded tokens
        sum_x = x.sum(dim=0)                               # B x D
        valid_counts = valid_mask.sum(dim=0).clamp(min=1e-6).unsqueeze(-1)  # B x 1
        x = sum_x / valid_counts                           # B x D (masked mean)

        logits = self.head(x)

        if return_loss and not self.flop_measure:
            return logits, boundary_loss, avg_boundaries_per_batch, boundary_ratio
        else:
            return logits


class SoftBoundaryPredictor(nn.Module):
    def __init__(self, d_model, d_inner, activation_function,
                 temp, prior_upper_bound, prior_lower_bound, bp_type, threshold=0.5,
                 image_size=None, patch_size=None, embed_dim=None):
        super().__init__()

        self.temp = temp
        self.prior_upper_bound = prior_upper_bound
        self.prior_lower_bound = prior_lower_bound
        self.bp_type = bp_type
        self.threshold = threshold
        self.embed_dim = embed_dim
        if image_size is not None and patch_size is not None:
            self.image_size = image_size
            self.patch_size = patch_size
            self.num_patches = (image_size // patch_size) ** 2

        if activation_function == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == 'gelu':
            activation_fn = torch.nn.GELU()

        self.boundary_predictor = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Linear(d_inner, 1),
        )

        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, hidden):
        # Hidden is of shape [seq_len x bs x d_model]
        # Boundaries we return are [bs x seq_len]

        boundary_logits = self.boundary_predictor(hidden).squeeze(-1).transpose(0, 1)
        boundary_probs = torch.sigmoid(boundary_logits)

        if self.bp_type == 'gumbel':
            bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=self.temp,
                probs=boundary_probs,
            )

            soft_boundaries = bernoulli.rsample()

            hard_boundaries = (soft_boundaries > self.threshold).float()
            hard_boundaries = (
                hard_boundaries - soft_boundaries.detach() + soft_boundaries
            )
        elif self.bp_type in ['entropy', 'unigram']:
            soft_boundaries = boundary_probs
            hard_boundaries = (soft_boundaries > self.threshold).float()

        return soft_boundaries, hard_boundaries

    def calc_loss(self, preds: torch.Tensor) -> torch.Tensor:
        """
        Penalize boundary rates only if outside the [prior_lower_bound, prior_upper_bound] interval.
        preds: B x T hard boundary tensor (0/1)
        """
        # Compute boundary rate per batch
        boundary_rate = preds.float().mean(dim=1)  # B

        # Compute penalty only outside the interval
        upper_violation = (boundary_rate - self.prior_upper_bound).clamp(min=0)
        lower_violation = (self.prior_lower_bound - boundary_rate).clamp(min=0)

        # Loss = mean squared deviation outside interval
        loss = (lower_violation + upper_violation).mean()
        return loss

    def calc_stats(self, preds, gt):
        # B x T
        preds, gt = preds.bool(), gt.bool()
        TP = ((preds == gt) & preds).sum().item()
        FP = ((preds != gt) & preds).sum().item()
        FN = ((preds != gt) & (~preds)).sum().item()

        acc = (preds == gt).sum().item() / gt.numel()

        if TP == 0:
            precision, recall = 0, 0
        else:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

        stats = {"acc": acc, "precision": precision, "recall": recall}

        return stats


class SoftDTPViT(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=(2, 8, 0),
                 num_heads=12,
                 mlp_ratio=4.0,
                 drop_rate=0.1,
                 attn_drop_rate=0.1,
                 temp=1.0,
                 compression_rate=(0.4, 0.6),
                 bp_type='gumbel',
                 threshold=0.5,
                 num_classes=1000,
                 activation_function='gelu',
                 flop_measure: bool = False,
        ):

        super().__init__()
        self.flop_measure = flop_measure
        self.prior = compression_rate
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.seq_len = self.num_patches

        # patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.dropout = nn.Dropout(drop_rate)

        # positional embedding
        self.pos_emb = PositionalEmbedding(embed_dim)
        self.r_w_bias = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))
        self.r_r_bias = nn.Parameter(torch.zeros(num_heads, embed_dim // num_heads))
        
        def create_decoder_layers(n_layers):
            layers = nn.ModuleList(
                [
                    RelPartialLearnableDecoderLayer(
                        n_head=num_heads,
                        d_model=embed_dim,
                        d_head=embed_dim // num_heads,
                        d_inner=int(embed_dim * mlp_ratio),
                        dropout=drop_rate,
                        dropatt=attn_drop_rate,
                        pre_lnorm=False,
                        activation_function=activation_function,
                    )
                    for _ in range(n_layers)
                ]
            )

            return layers

        # pre-pooling block
        self.pre_blocks = create_decoder_layers(depth[0])

        # post-pooling block
        self.short_blocks = create_decoder_layers(depth[1])

        self.prior_lower_bound = compression_rate[0]
        self.prior_upper_bound = compression_rate[1]

        # boundary predictor
        self.boundary_predictor = SoftBoundaryPredictor(
            d_model=embed_dim,
            d_inner=int(embed_dim * mlp_ratio),
            activation_function=activation_function,
            prior_lower_bound=self.prior_lower_bound,
            prior_upper_bound=self.prior_upper_bound,
            temp=temp,
            bp_type=bp_type,
            threshold=threshold
        )

        # layer normalization and null token
        self.down_ln = nn.LayerNorm(embed_dim)
        self.null_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.null_token, std=0.02)

        # final projection
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward_after_pooling_with_attn_masks(self, core_input: torch.Tensor, layers, attention_mask: torch.Tensor):
        """
        Process input with relative attention and padding-aware masking.
        """
        T, _, _ = core_input.size()

        # Compute position embeddings
        pos_seq = torch.arange(T - 1, -1, -1.0, device=core_input.device, dtype=core_input.dtype)
        pos_emb = self.pos_emb(pos_seq)
        pos_emb = self.dropout(pos_emb)

        core_out = core_input
        for layer in layers:
            core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask=attention_mask)
        return core_out

    def encode(self, x: torch.Tensor, return_loss: bool = False):
        """
        Encode input image to feature sequence without final pooling.
        Returns:
            features OR (features, boundary_loss, avg_boundaries, boundary_ratio)
        """
        B = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)                  # B x C x H' x W'
        x = x.flatten(2).transpose(1, 2)         # B x L x C
        x = self.dropout(x)                      # B x L x C

        # Positional embedding (for pre-blocks)
        pos_seq = torch.arange(self.seq_len - 1, -1, -1.0,
                            device=x.device, dtype=x.dtype)
        r = self.pos_emb(pos_seq)                # L x 1 x C

        # Pre-pooling transformer blocks
        x = x.transpose(0, 1)                    # L x B x C
        for block in self.pre_blocks:
            x = block(x, r, self.r_w_bias, self.r_r_bias)

        # boundary prediction
        if self.flop_measure:
            # Simulate hard boundaries for FLOP measurement
            L = x.size(0)
            num_tokens_to_keep = max(1, int(L * (self.prior_lower_bound + self.prior_upper_bound) / 2))
            indices = torch.linspace(0, L - 1, steps=num_tokens_to_keep).round().long()
            hard_boundaries = torch.zeros(B, L, device=x.device)
            hard_boundaries[:, indices] = 1
        else:
            _, hard_boundaries = self.boundary_predictor(x)  # B x L

        # Downsampling (Dynamic Token Pooling)
        hidden = self.down_ln(x)               # L x B x D
        shortened_hidden = downsample(
            boundaries=hard_boundaries,
            hidden=hidden,
            null_group=self.null_token
        )                                        # S x B x D

        # attention mask for post-pooling transformer layers
        S = shortened_hidden.size(0)
        pad_mask = shortened_hidden.abs().sum(-1).eq(0)       # S x B (1 where padded, 0 where regular)

        attn_mask = pad_mask.transpose(0, 1).unsqueeze(1)     # B x 1 x S
        attn_mask = attn_mask.expand(B, S, S)                 # B x S x S

        # post-pooling transformer blocks
        shortened_hidden = self.forward_after_pooling_with_attn_masks(
            shortened_hidden,
            self.short_blocks,
            attention_mask=attn_mask
        )

        # return features and optional loss
        features = shortened_hidden  # S x B x D

        if return_loss and not self.flop_measure:
            # Binomial boundary loss (no need for mask since all sequences have the same number of tokens)
            boundary_loss = self.boundary_predictor.calc_loss(hard_boundaries)
            avg_boundaries_per_batch = hard_boundaries.sum(dim=1).float().mean().item()
            boundary_ratio = avg_boundaries_per_batch / hard_boundaries.size(1)
            return features, boundary_loss, avg_boundaries_per_batch, boundary_ratio
        else:
            return features

    def forward(self, x, return_loss=False):
        """
        Full forward pass including pooling to class logits.
        """
        features_out = self.encode(x, return_loss=return_loss)

        if return_loss and not self.flop_measure:
            # encode returns tuple (features, loss, avg_boundaries, boundary_ratio)
            x, boundary_loss, avg_boundaries_per_batch, boundary_ratio = features_out
        else:
            x = features_out

        # pool across sequence dimension with mean pooling
        pad_mask = x.abs().sum(-1).eq(0).float()           # S x B
        valid_mask = 1.0 - pad_mask                        # S x B
        valid_mask_exp = valid_mask.unsqueeze(-1)          # S x B x 1

        x = x * valid_mask_exp                             # Mask padded tokens
        sum_x = x.sum(dim=0)                               # B x D
        valid_counts = valid_mask.sum(dim=0).clamp(min=1e-6).unsqueeze(-1)  # B x 1
        x = sum_x / valid_counts                           # B x D (masked mean)

        logits = self.head(x)

        if return_loss and not self.flop_measure:
            return logits, boundary_loss, avg_boundaries_per_batch, boundary_ratio
        else:
            return logits







###############################################################################################################
##################################### VIT backbone ########################################
###############################################################################################################
class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_chans: int = 3, embed_dim: int = 768):
        """
        Patch Embedding Layer
        Args:
            image_size (int): Size of the input image (assumed square).
            patch_size (int): Size of each patch (assumed square).
            in_chans (int): Number of input channels (e.g., 3 for RGB).
            embed_dim (int): Dimension of the embedding space.
        """
        super().__init__()
        self.img_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size // patch_size, image_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        """
            input: [batch size, # channels, height, width]
            output: [batch size, # patches, embed_dim]
        """
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class TransformerBlock(nn.Module):
    """
    A single Transformer block with multi-head self-attention and MLP layers.
    Args:
        dim (int): Dimension of the input features.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
        drop (float): Dropout rate applied after attention and MLP layers.
        attn_drop (float): Dropout rate applied within attention layers.
        activation_function (str): Activation function used in MLP layers ('gelu' or 'relu').
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0., attn_drop=0., activation_function='gelu'):
        
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=False)
        self.drop_path = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)

        act_fn = nn.GELU() if activation_function == 'gelu' else nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            act_fn,
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

####################################
### more baselines #################
####################################

class PatchMergeAvg(nn.Module):
    """
    Fixed 2x2 spatial downsampling WITHOUT channel concat/projection.
    Input:  x (B, L, C) with L = H*W
    Output: (B, (H/2)*(W/2), C)
    """
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution  # (H, W)
        self.dim = dim

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W, f"Expected L=H*W, got L={L}, H*W={H*W}"
        assert H % 2 == 0 and W % 2 == 0, f"H and W must be even, got {H},{W}"

        x = x.view(B, H, W, C)
        x00 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x10 = x[:, 1::2, 0::2, :]
        x01 = x[:, 0::2, 1::2, :]
        x11 = x[:, 1::2, 1::2, :]
        x = (x00 + x10 + x01 + x11) * 0.25
        x = x.reshape(B, (H // 2) * (W // 2), C)
        return x

class TokenPool1DAvg(nn.Module):
    """
    Fixed 1-D token pooling (non-spatial-aware).
    Pools consecutive tokens in the sequence by averaging groups of `pool`.
    
    Input:  x (B, L, C)
    Output: (B, L/pool, C)
    """
    def __init__(self, pool: int = 4, handle_tail: str = "trim"):
        """
        Args:
            pool: group size for averaging. Use 4 to mimic 2x2 spatial merge.
            handle_tail:
                - "trim": drop leftover tokens if L % pool != 0
                - "pad":  right-pad with zeros to the next multiple of `pool`
        """
        super().__init__()
        assert handle_tail in ("trim", "pad")
        self.pool = pool
        self.handle_tail = handle_tail

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        p = self.pool

        if L % p != 0:
            if self.handle_tail == "trim":
                L_new = (L // p) * p
                x = x[:, :L_new, :]
            else:  # pad
                L_new = ((L + p - 1) // p) * p
                pad_len = L_new - L
                pad = x.new_zeros(B, pad_len, C)
                x = torch.cat([x, pad], dim=1)

        # now length is divisible by p
        B, L, C = x.shape
        x = x.view(B, L // p, p, C).mean(dim=2)
        return x

class SingleAdaptedSwin(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,            # constant across pre/post
                 depth=(2, 8),             # (pre_depth, post_depth)
                 num_heads=(12, 12),       # (pre_heads, post_heads) or single int
                 mlp_ratio=4.0,
                 drop_rate=0.1,
                 num_classes=1000,
                 activation_function='gelu',
                 flop_measure: bool = False,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.flop_measure = flop_measure
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        C = embed_dim

        # ----- grids -----
        Hp = image_size // patch_size
        Wp = image_size // patch_size
        assert Hp * patch_size == image_size and Wp * patch_size == image_size
        assert (Hp % 2 == 0) and (Wp % 2 == 0), "Need even H/ W (one 2x merge)."
        self.grid0 = (Hp, Wp)
        self.grid1 = (Hp // 2, Wp // 2)

        L0 = Hp * Wp
        L1 = (Hp // 2) * (Wp // 2)

        # ----- heads -----
        if isinstance(num_heads, int):
            heads = (num_heads, num_heads)
        else:
            assert len(num_heads) == 2
            heads = num_heads

        # ----- patch embed -----
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_chans, C)
        self.dropout = nn.Dropout(drop_rate)

        # ----- pos embeddings (both C, constant width) -----
        self.pos_pre  = nn.Parameter(torch.zeros(1, 1 + L0, C))
        self.pos_post = nn.Parameter(torch.zeros(1, 1 + L1, C))
        nn.init.trunc_normal_(self.pos_pre,  std=0.02)
        nn.init.trunc_normal_(self.pos_post, std=0.02)

        # ----- transformer stacks -----
        def make_layers(n_layers, d_model, nhead):
            return nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=int(d_model * mlp_ratio),
                    dropout=drop_rate,
                    activation=activation_function,
                    batch_first=False,
                    norm_first=True
                )
                for _ in range(max(0, n_layers))
            ])

        self.pre_blocks  = make_layers(depth[0], C, heads[0])
        self.post_blocks = make_layers(depth[1], C, heads[1])

        # ----- fixed pooling (avg) -----
        self.merge = PatchMergeAvg(self.grid0, dim=C)  # L0 -> L1, keep C

        # ----- norm + head -----
        self.post_ln = norm_layer(C)
        self.head    = nn.Linear(C, num_classes)

    # utils
    def _add_pos(self, x, pos_param):
        B, L, C = x.shape
        pos = pos_param[:, 1:1+L, :].to(device=x.device, dtype=x.dtype)
        return x + pos

    def _run_stack(self, x, layers):
        for blk in layers:
            x = blk(x)
        return x

    def encode(self, x: torch.Tensor, return_loss: bool = False):
        # pre
        x = self.patch_embed(x)                 # (B, L0, C)
        x = self.dropout(x)
        x = self._add_pos(x, self.pos_pre)
        x = x.transpose(0, 1)                   # (L0, B, C)
        x = self._run_stack(x, self.pre_blocks) # (L0, B, C)

        # merge (no channel change)
        x = x.transpose(0, 1)                   # (B, L0, C)
        x = self.merge(x)                       # (B, L1, C)
        x = self._add_pos(x, self.pos_post)
        x = x.transpose(0, 1)                   # (L1, B, C)

        # post
        x = self._run_stack(x, self.post_blocks)  # (L1, B, C)

        if return_loss:
            # keep DTPViT API
            dummy_loss = torch.zeros([], device=x.device)
            # single fixed merge → tokens /4
            avg_boundaries = 0.0
            boundary_ratio = 1.0 / 4.0
            return x, dummy_loss, avg_boundaries, boundary_ratio
        else:
            return x

    def forward(self, x, return_loss: bool = False):
        out = self.encode(x, return_loss=return_loss)
        if return_loss:
            feats, boundary_loss, avg_boundaries, boundary_ratio = out
        else:
            feats = out

        # mean pool over sequence (dense)
        x = feats.mean(dim=0)   # (B, C)
        x = self.post_ln(x)
        logits = self.head(x)

        if return_loss:
            return logits, boundary_loss, avg_boundaries, boundary_ratio
        else:
            return logits
