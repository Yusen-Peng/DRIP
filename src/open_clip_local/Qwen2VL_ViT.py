import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Callable
from torch.nn import LayerNorm
from typing import Tuple
from transformers import AutoModel
from transformers.models.siglip.modeling_siglip import SiglipModel

from .BP import BoundaryPredictor, downsample_with_indices


class Qwen2VLVisionConfig:
    model_type = "qwen2_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=32,
        embed_dim=1280,
        hidden_size=3584,
        hidden_act="quick_gelu",
        mlp_ratio=4,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        initializer_range=0.02,
        output_dim=512,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.initializer_range = initializer_range
        self.output_dim = output_dim

class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(1.702 * input)

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class VisionMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        hidden_act = hidden_act
        self.act = QuickGELUActivation()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states

"""
    2D-ROPE part implementation.
"""

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class VisionAttention(nn.Module):
    def __init__(self, config: Qwen2VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward 
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
        ]

        attn_outputs = [
            attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config: Qwen2VLVisionConfig, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = LayerNorm(config.embed_dim, eps=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)

        self.attn = VisionAttention(config=config)
        self.mlp = VisionMlp(dim=config.embed_dim, hidden_dim=mlp_hidden_dim, hidden_act=config.hidden_act)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

"""
    Qwen2VL version of ViT (with 2D-ROPE).
"""

class SiglipVisionHead(nn.Module):
    """
    SigLIP vision_model.head:
      probe + MHA pooling + LN + MLP(residual)
    Output: (B, embed_dim)
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, num_probes: int = 1):
        super().__init__()
        self.probe = nn.Parameter(torch.zeros(num_probes, embed_dim))  # (P, D)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-6)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            QuickGELUActivation(),
            nn.Linear(hidden_dim, embed_dim),   # <-- MUST be embed_dim for residual
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, L, D = tokens.shape
        q = self.probe.unsqueeze(0).expand(B, -1, -1)  # (B, P, D)

        pooled, _ = self.attention(q, tokens, tokens, need_weights=False)  # (B, P, D)
        pooled = pooled.mean(dim=1)  # (B, D)

        x = self.layernorm(pooled)
        x = pooled + self.mlp(x)     # residual is valid now
        return x                     # (B, embed_dim)
    

class Qwen2VLViT(nn.Module):
    config: Qwen2VLVisionConfig
    input_modalities = ("image", "video")
    _no_split_modules = ["Qwen2VLVisionBlock"]
    _input_embed_layer = "patch_embed"

    def __init__(self, config: Qwen2VLVisionConfig) -> None:
        super().__init__()
        self.config = config  # keep it for reference
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )
        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Qwen2VLVisionBlock(config) for _ in range(config.depth)])
        self.gradient_checkpointing = False
        self.pool_type = 'avg'  # 'avg' | 'tok' | 'none'
        self.final_ln_after_pool = True
        self.ln_post = LayerNorm(config.embed_dim, eps=1e-6)
        self.attn_pool = None
        self.attn_pool_contrastive = None
        self.attn_pool_type = 'parallel'
        self.output_dim = config.output_dim
        self.proj = nn.Parameter(torch.randn(config.embed_dim, self.output_dim))

        self.use_siglip_head = False # set to False by default
        self.siglip_head = SiglipVisionHead(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_probes=1,
        )


    def get_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks[0].mlp.fc2.weight.device

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def encode(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width dimensions of feature shape for each image. Each row contains [t, h, w] values.
        """
        # hidden_states: (B, 3, H, W)
        B, _, H, W = hidden_states.shape
        # initialize grid_thw parameter
        gh, gw = H // self.config.patch_size, W // self.config.patch_size
        grid_thw = hidden_states.new_empty((B, 3), dtype=torch.long)
        grid_thw[:, 0] = 1
        grid_thw[:, 1] = gh
        grid_thw[:, 2] = gw

        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )        
        # reshape (B*L, D) to (B, L, D)
        reshaped_hidden_states = hidden_states.view(B, gh * gw, self.config.embed_dim)
        return reshaped_hidden_states
    
    def _global_pool(self, x: torch.Tensor):
        if self.pool_type == 'avg':
            pooled, tokens = x.mean(dim=1), x
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x
        return pooled, tokens

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

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        tokens = self.encode(hidden_states, **kwargs)  # (B, L, D)

        # SigLIP does post_layernorm before head pooling
        if self.use_siglip_head and self.siglip_head is not None:
            tokens_ln = self.ln_post(tokens)
            pooled = self.siglip_head(tokens_ln)  # (B, D)
            # now map to OpenCLIP embed space (output_dim)
            pooled = pooled @ self.proj                # (B, output_dim)
            # pooled = F.normalize(pooled, dim=-1)
            return pooled

        pooled, _ = self._pool(tokens)
        pooled = pooled @ self.proj
        return pooled
    

    @torch.no_grad()
    def load_from_qwen2vl_checkpoint(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        verbose: bool = True,
    ):
        dtype = self.get_dtype()
        device = self.get_device()
        hf = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map={"": "cpu"},  # keep CPU to avoid random GPU OOM while loading 7B
        )

        # grab vision tower (Qwen2-VL uses `visual`)
        if hasattr(hf, "model") and hasattr(hf.model, "visual") and hf.model.visual is not None:
            hf_visual = hf.model.visual
            prefix = ""
        else:
            raise AttributeError("Could not find vision tower at `model.visual` or `model.model.visual`.")

        hf_sd = hf_visual.state_dict()
        self.to(device=device, dtype=dtype)

        drop_prefixes = (
            "siglip_head.",  # yours
        )

        filtered = {}
        for k, v in hf_sd.items():
            # HF key -> your key mapping
            # Most common case: keys already match (patch_embed.*, blocks.*, ln_post.*)
            nk = k

            # If HF ever saves with a "visual." prefix (sometimes happens when you use hf.state_dict()),
            # you can uncomment this:
            # if nk.startswith("visual."): nk = nk[len("visual."):]

            if any(nk.startswith(dp) for dp in drop_prefixes):
                continue

            filtered[nk] = v.to(device=device, dtype=dtype)

        # 5) Load (strict=False = we’re chill about extra/missing keys)
        missing, unexpected = self.load_state_dict(filtered, strict=False)

        if verbose:
            print(f"[load_from_qwen2vl_checkpoint] loaded from {model_name}")
            print(f"  missing keys: {len(missing)}")
            print(f"  unexpected keys: {len(unexpected)}")
            # helpful to sanity check what didn't load
            if len(missing) and len(missing) < 40:
                print("  missing:", missing)
            if len(unexpected) and len(unexpected) < 40:
                print("  unexpected:", unexpected)

        return {"missing": missing, "unexpected": unexpected}








    @torch.no_grad()
    def load_siglip2_vision_from_full_sd(self, verbose: bool = True):
        """
        Load SigLIP2 vision tower from HF SiglipModel.state_dict() into this RoPE ViT.

        Assumes this model matches:
          depth=12, embed_dim=768, num_heads=12, patch_size=16, temporal_patch_size=1
        """

        # load the checkpoint
        ckpt = "google/siglip2-base-patch16-224"
        model: SiglipModel = AutoModel.from_pretrained(ckpt, trust_remote_code=True)
        full_sd = model.state_dict()

        # report on how many tensors were loaded, skipped (e.g. probe head), missing, or mismatched in shape
        rep = {"loaded": [], "skipped": [], "missing": [], "mismatch": []}

        """
            helper functions.
        """
        def log(msg):
            if verbose:
                print(msg)
        
        def copy_(dst: torch.Tensor, src: torch.Tensor, name: str):
            
            if tuple(dst.shape) != tuple(src.shape):
                rep["mismatch"].append(f"{name}: dst{tuple(dst.shape)} != src{tuple(src.shape)}")
                return
            
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
            rep["loaded"].append(name)


        """
        component: patch embedding
            SigLIP2: Conv2d weight (D, 3, ps, ps) + bias (D)
            Qwen2VLViT:  Conv3d weight (D, 3, tp, ps, ps) and bias=False
        """
        pe_w = "vision_model.embeddings.patch_embedding.weight"
        pe_b = "vision_model.embeddings.patch_embedding.bias"
        if pe_w in full_sd:
            src_w = full_sd[pe_w]
            dst_w = self.patch_embed.proj.weight
            if src_w.ndim == 4 and dst_w.ndim == 5:
                tp = dst_w.shape[2]
                if tp != 1:
                    inflated = src_w.unsqueeze(2).repeat(1, 1, tp, 1, 1) / tp
                else:
                    inflated = src_w.unsqueeze(2)  # (D,3,1,ps,ps)
                copy_(dst_w, inflated, "patch_embed.proj.weight")
            else:
                copy_(dst_w, src_w, "patch_embed.proj.weight")
        else:
            rep["missing"].append(pe_w)
        if pe_b in full_sd:
            if getattr(self.patch_embed.proj, "bias", None) is None:
                rep["skipped"].append(f"{pe_b}")
            else:
                copy_(self.patch_embed.proj.bias, full_sd[pe_b], "patch_embed.proj.bias")
        else:
            rep["missing"].append(pe_b)

        """
            skip APE from SigLIP2 checkpoint
        """
        pos_k = "vision_model.embeddings.position_embedding.weight"
        if pos_k in full_sd:
            rep["skipped"].append(pos_k)


        """
            load SigLIP vision head
        """
        # infer num_probes from ckpt tensor shape
        probe_k = "vision_model.head.probe"
        if probe_k in full_sd:
            probe: torch.Tensor = full_sd[probe_k]  # shape likely (P, D) or (1, P, D) depending on impl
            if probe.ndim == 3:
                probe = probe.squeeze(0)
            P, D = probe.shape

            # NOTE: rebuild head with correct P if needed
            if self.siglip_head.probe.shape[0] != P:
                self.siglip_head = SiglipVisionHead(
                    embed_dim=self.config.embed_dim,
                    output_dim=self.config.output_dim,
                    num_heads=self.config.num_heads,
                    mlp_ratio=self.config.mlp_ratio,
                    num_probes=P,
                ).to(device=self.get_device(), dtype=self.get_dtype())

            copy_(self.siglip_head.probe, probe, "siglip_head.probe")
        else:
            rep["missing"].append(probe_k)

        # attention weights (MultiheadAttention uses same parameter names)
        for name in ["in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias"]:
            ck = f"vision_model.head.attention.{name}"
            if ck in full_sd:
                # map to pytorch MHA params
                if name.startswith("out_proj."):
                    attr = name.split(".", 1)[1]  # "weight" or "bias"
                    copy_(getattr(self.siglip_head.attention.out_proj, attr), full_sd[ck], f"siglip_head.attention.{name}")
                else:
                    copy_(getattr(self.siglip_head.attention, name), full_sd[ck], f"siglip_head.attention.{name}")
            else:
                rep["missing"].append(ck)

        # layernorm
        for p in ["weight", "bias"]:
            ck = f"vision_model.head.layernorm.{p}"
            if ck in full_sd:
                copy_(getattr(self.siglip_head.layernorm, p), full_sd[ck], f"siglip_head.layernorm.{p}")
            else:
                rep["missing"].append(ck)

        # mlp
        for fc in ["fc1", "fc2"]:
            for p in ["weight", "bias"]:
                ck = f"vision_model.head.mlp.{fc}.{p}"
                if ck in full_sd:
                    # our mlp is Sequential: [Linear, Act, Linear]
                    mod = self.siglip_head.mlp[0] if fc == "fc1" else self.siglip_head.mlp[2]
                    copy_(getattr(mod, p), full_sd[ck], f"siglip_head.mlp.{fc}.{p}")
                else:
                    rep["missing"].append(ck)

        # enable it
        self.use_siglip_head = True

        """
            transformer blocks.
        """
        for i, blk in enumerate(self.blocks):
            base = f"vision_model.encoder.layers.{i}"
            blk: Qwen2VLVisionBlock = blk

            # norm1 and norm2
            for ln_name, ln_mod in [("layer_norm1", blk.norm1), ("layer_norm2", blk.norm2)]:
                for p in ["weight", "bias"]:
                    k = f"{base}.{ln_name}.{p}"
                    if k in full_sd:
                        copy_(getattr(ln_mod, p), full_sd[k], f"blocks.{i}.{ln_name}.{p}")
                    else:
                        rep["missing"].append(k)

            # mlp
            for fc in ["fc1", "fc2"]:
                for p in ["weight", "bias"]:
                    k = f"{base}.mlp.{fc}.{p}"
                    if k in full_sd:
                        copy_(getattr(getattr(blk.mlp, fc), p), full_sd[k], f"blocks.{i}.mlp.{fc}.{p}")
                    else:
                        rep["missing"].append(k)

            # attention
            # SigLIP2 has separate q/k/v proj; Qwen2VLViT has fused qkv
            q_w = full_sd.get(f"{base}.self_attn.q_proj.weight", None)
            k_w = full_sd.get(f"{base}.self_attn.k_proj.weight", None)
            v_w = full_sd.get(f"{base}.self_attn.v_proj.weight", None)
            q_b = full_sd.get(f"{base}.self_attn.q_proj.bias", None)
            k_b = full_sd.get(f"{base}.self_attn.k_proj.bias", None)
            v_b = full_sd.get(f"{base}.self_attn.v_proj.bias", None)

            if q_w is None or k_w is None or v_w is None:
                rep["missing"].append(f"{base}.self_attn.(q/k/v)_proj.weight")
            else:
                # fusion happens: (3D, D)
                fused_w = torch.cat([q_w, k_w, v_w], dim=0)  
                copy_(blk.attn.qkv.weight, fused_w, f"blocks.{i}.attn.qkv.weight")

            if q_b is None or k_b is None or v_b is None:
                rep["missing"].append(f"{base}.self_attn.(q/k/v)_proj.bias")
            else:
                fused_b = torch.cat([q_b, k_b, v_b], dim=0)  # (3D,)
                copy_(blk.attn.qkv.bias, fused_b, f"blocks.{i}.attn.qkv.bias")

            # out proj
            out_w = f"{base}.self_attn.out_proj.weight"
            out_b = f"{base}.self_attn.out_proj.bias"
            if out_w in full_sd:
                copy_(blk.attn.proj.weight, full_sd[out_w], f"blocks.{i}.attn.proj.weight")
            else:
                rep["missing"].append(out_w)
            if out_b in full_sd:
                copy_(blk.attn.proj.bias, full_sd[out_b], f"blocks.{i}.attn.proj.bias")
            else:
                rep["missing"].append(out_b)

        """
            final layer normalization layers.
        """

        for p in ["weight", "bias"]:
            k = f"vision_model.post_layernorm.{p}"
            if k in full_sd:
                copy_(getattr(self.ln_post, p), full_sd[k], f"ln_post.{p}")
            else:
                rep["missing"].append(k)

        log(f"[🥹🥹🥹SigLIP2 vision -> Qwen2VL] loaded={len(rep['loaded'])} "
            f"mismatch={len(rep['mismatch'])} missing={len(rep['missing'])} skipped={len(rep['skipped'])}")




"""
    Qwen2VL version of DRIP (with 2D-ROPE).
"""


class Qwen2VLDRIP(nn.Module):
    config: Qwen2VLVisionConfig
    input_modalities = ("image", "video")
    _no_split_modules = ["Qwen2VLVisionBlock"]
    _input_embed_layer = "patch_embed"

    def __init__(self, 
        config: Qwen2VLVisionConfig,
        depth: tuple[int],
        compression_rate: float,
        temp: float,
        flop_measure: bool = False,
        threshold: float = 0.5
        ) -> None:
        super().__init__()
        self.config = config  # keep it for reference
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )
        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks_pre = nn.ModuleList([Qwen2VLVisionBlock(config) for _ in range(depth[0])])
        self.blocks_post = nn.ModuleList([Qwen2VLVisionBlock(config) for _ in range(depth[1])])

        self.prior = compression_rate
        self.boundary_predictor = BoundaryPredictor(
            d_model=config.embed_dim,
            d_inner=int(config.embed_dim * config.mlp_ratio),
            activation_function="gelu",
            temp=temp,
            prior=compression_rate,
            bp_type='gumbel',
            threshold=threshold
        )

        self.gradient_checkpointing = False
        self.pool_type = 'avg'  # 'avg' | 'tok' | 'none'
        self.final_ln_after_pool = True
        self.ln_post = LayerNorm(config.embed_dim, eps=1e-6)
        self.attn_pool = None
        self.attn_pool_contrastive = None
        self.attn_pool_type = 'parallel'
        self.output_dim = config.embed_dim
        self.proj = nn.Parameter(torch.randn(config.embed_dim, self.output_dim))
        self.flop_measure = flop_measure
        self.null_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        nn.init.normal_(self.null_token, std=0.02)


    def get_dtype(self) -> torch.dtype:
        return self.blocks_pre[0].mlp.fc2.weight.dtype

    def get_device(self) -> torch.device:
        return self.blocks_pre[0].mlp.fc2.weight.device

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def encode(
        self,
        hidden_states: torch.Tensor,
        return_loss: bool,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        grid_thw (`torch.LongTensor` of shape `(num_images, 3)`):
            The temporal, height and width dimensions of feature shape for each image. Each row contains [t, h, w] values.
        """
        # hidden_states: (B, 3, H, W)
        B, _, H, W = hidden_states.shape
        # initialize grid_thw parameter
        gh, gw = H // self.config.patch_size, W // self.config.patch_size
        grid_thw = hidden_states.new_empty((B, 3), dtype=torch.long)
        grid_thw[:, 0] = 1
        grid_thw[:, 1] = gh
        grid_thw[:, 2] = gw

        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks_pre:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        # reshape (B*L, D) to (B, L, D)
        reshaped_hidden_states = hidden_states.view(B, gh * gw, self.config.embed_dim)
        # print(f"🥎🥎🥎shape check: {reshaped_hidden_states.shape} - should be (B, L, D).")
        
        B, L, _ = reshaped_hidden_states.shape
        if self.flop_measure:
            num_tokens_to_keep = max(1, int(L * self.prior))
            indices = torch.linspace(0, L - 1, steps=num_tokens_to_keep).round().long()
            hard_boundaries = torch.zeros(B, L, device=reshaped_hidden_states.device)
            # hard boundaries: [B, L]
            hard_boundaries[:, indices] = 1
        else:
            x_transposed = reshaped_hidden_states.transpose(0, 1) # [B, L, D] -> [L, B, D]
            # hard boundaries: [B, L]
            _, hard_boundaries = self.boundary_predictor(x_transposed) # input is [L, B, D]

        hidden = reshaped_hidden_states.transpose(0, 1) # [B, L, D] -> [L, B, D]
        # print(f"👔👔👔👔shape before downsample: {hidden.shape}👔👔👔👔; should be (L, B, D)")
        shortened_hidden, rep_idx = downsample_with_indices(
            boundaries=hard_boundaries,
            hidden=hidden,
            null_group=self.null_token
        ) # [L, B, D] -> [S, B, D]
        shortened_hidden = shortened_hidden.transpose(0, 1) # [S, B, D] -> [B, S, D]
        S = shortened_hidden.size(1)

        """
            rebuild cu_seqlens for post blocks: each sample has S tokens.
        """
        cu_seqlens_post = torch.arange(
            0, (B + 1) * S, step=S,
            device=shortened_hidden.device,
            dtype=torch.int32 if not torch.jit.is_tracing() else grid_thw.dtype,
        )

        """
            prepare position embeddings for post blocks.
        """
        offset = (torch.arange(B, device=rep_idx.device) * L)[:, None]
        rep_global = (rep_idx + offset).reshape(-1)    
        emb_post = emb[rep_global]                
        position_embeddings_post = (emb_post.cos(), emb_post.sin())
        
        reshaped_shortened_hidden_states = shortened_hidden.reshape(B * S, self.config.embed_dim) # reshape back to (B*S, D)
        
        for blk in self.blocks_post:
            reshaped_shortened_hidden_states = blk(
                reshaped_shortened_hidden_states,
                cu_seqlens=cu_seqlens_post,
                position_embeddings=position_embeddings_post,
                **kwargs,
            )
        feature_out = reshaped_shortened_hidden_states.view(B, S, self.config.embed_dim) # reshape (B*S, D) to (B, S, D)

        if return_loss and not self.flop_measure:
            boundary_loss = self.boundary_predictor.calc_loss(hard_boundaries)
            avg_boundaries_per_batch = hard_boundaries.sum(dim=1).float().mean().item()
            boundary_ratio = avg_boundaries_per_batch / hard_boundaries.size(1)
            return feature_out, boundary_loss, avg_boundaries_per_batch, boundary_ratio
        else:
            return feature_out # [B, S, D]
    
    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens


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

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_loss: bool = False
    ) -> torch.Tensor:
        features_out = self.encode(hidden_states, return_loss=return_loss) # (B, L, D)

        if return_loss and not self.flop_measure:
            # encode returns tuple (features, loss, avg_boundaries, boundary_ratio)
            tensor, boundary_loss, avg_boundaries_per_batch, boundary_ratio = features_out
        else:
            tensor = features_out
        
        pooled, _ = self._pool(tensor) # [B, L, D] -> [B, D]
        pooled = pooled @ self.proj # [B, D] -> [B, output_dim]


        if return_loss and not self.flop_measure:
            return pooled, boundary_loss, avg_boundaries_per_batch, boundary_ratio # [B, output_dim]
        else:
            return pooled # [B, output_dim]

