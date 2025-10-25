# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from .kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


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


class HierarchicalAdaptedSwin(nn.Module):
    """
    Hierarchical Swin-like FIXED pooling with CONSTANT channel dim:
      Stage 0: (B, L0, C)
      Merge -> Stage 1: (B, L1=L0/4, C)
      Merge -> Stage 2: (B, L2=L0/16, C)
      Merge -> Stage 3: (B, L3=L0/64, C)
      Mean pool -> head

    This is a fair fixed-pooling baseline vs DRIP: token count shrinks; channel stays the same.
    """
    def __init__(self,
                 image_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,                 # constant across stages
                 depth=(2, 2, 6, 2),
                 num_heads=6,                  # can be int or a 4-tuple; default keeps head_dim constant
                 mlp_ratio=4.0,
                 drop_rate=0.1,
                 num_classes=1000,
                 activation_function='gelu',
                 flop_measure: bool = False,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.flop_measure = flop_measure
        self.num_classes = num_classes

        # ----- grid sizes -----
        Hp = image_size // patch_size
        Wp = image_size // patch_size
        assert Hp * patch_size == image_size and Wp * patch_size == image_size
        assert (Hp % 8 == 0) and (Wp % 8 == 0), "Hp,Wp must be divisible by 8 (3 merges)."
        self.grid0 = (Hp, Wp)
        self.grid1 = (Hp // 2, Wp // 2)
        self.grid2 = (Hp // 4, Wp // 4)
        self.grid3 = (Hp // 8, Wp // 8)

        L0 = Hp * Wp
        L1 = self.grid1[0] * self.grid1[1]
        L2 = self.grid2[0] * self.grid2[1]
        L3 = self.grid3[0] * self.grid3[1]

        C = embed_dim   # constant across stages

        # ----- patch embed -----
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_chans, C)
        self.dropout = nn.Dropout(drop_rate)

        # ----- positional embeddings per stage (all C) -----
        self.pos0 = nn.Parameter(torch.zeros(1, 1 + L0, C))
        self.pos1 = nn.Parameter(torch.zeros(1, 1 + L1, C))
        self.pos2 = nn.Parameter(torch.zeros(1, 1 + L2, C))
        self.pos3 = nn.Parameter(torch.zeros(1, 1 + L3, C))
        for p in [self.pos0, self.pos1, self.pos2, self.pos3]:
            nn.init.trunc_normal_(p, std=0.02)

        # ----- heads per stage -----
        if isinstance(num_heads, int):
            heads = (num_heads, num_heads, num_heads, num_heads)
        else:
            assert len(num_heads) == 4
            heads = num_heads

        # ----- helper: encoder layer stacks -----
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
                ) for _ in range(max(0, n_layers))
            ])

        # ----- stage blocks (all at dim=C) -----
        self.blocks0 = make_layers(depth[0], C, heads[0])
        self.blocks1 = make_layers(depth[1], C, heads[1])
        self.blocks2 = make_layers(depth[2], C, heads[2])
        self.blocks3 = make_layers(depth[3], C, heads[3])

        # ----- fixed pooling (avg) between stages (preserve C) -----
        self.merge0 = PatchMergeAvg(self.grid0, dim=C)  # L0 -> L1, C
        self.merge1 = PatchMergeAvg(self.grid1, dim=C)  # L1 -> L2, C
        self.merge2 = PatchMergeAvg(self.grid2, dim=C)  # L2 -> L3, C

        # ----- norm + head -----
        self.norm3 = norm_layer(C)
        self.head  = nn.Linear(C, num_classes)

    # --- utils ---
    def _add_pos(self, x, pos_param):
        B, L, C = x.shape
        pos = pos_param[:, 1:1+L, :].to(device=x.device, dtype=x.dtype)
        return x + pos

    def _run_stack(self, x, layers):
        for blk in layers:
            x = blk(x)
        return x

    # --- encode through 4 stages, 3 merges ---
    def encode(self, x: torch.Tensor, return_loss: bool = False):
        # Stage 0
        x = self.patch_embed(x)              # (B, L0, C)
        x = self.dropout(x)
        x = self._add_pos(x, self.pos0)
        x = x.transpose(0, 1)                # (L0, B, C)
        x = self._run_stack(x, self.blocks0)

        # Merge 0
        x = x.transpose(0, 1)                # (B, L0, C)
        x = self.merge0(x)                   # (B, L1, C)
        x = self._add_pos(x, self.pos1)
        x = x.transpose(0, 1)                # (L1, B, C)
        x = self._run_stack(x, self.blocks1)

        # Merge 1
        x = x.transpose(0, 1)                # (B, L1, C)
        x = self.merge1(x)                   # (B, L2, C)
        x = self._add_pos(x, self.pos2)
        x = x.transpose(0, 1)                # (L2, B, C)
        x = self._run_stack(x, self.blocks2)

        # Merge 2
        x = x.transpose(0, 1)                # (B, L2, C)
        x = self.merge2(x)                   # (B, L3, C)
        x = self._add_pos(x, self.pos3)
        x = x.transpose(0, 1)                # (L3, B, C)
        x = self._run_stack(x, self.blocks3) # (L3, B, C)

        if return_loss:
            dummy_loss = torch.zeros([], device=x.device)
            cum_avg_boundaries = 0.0
            cum_boundary_ratio = 1.0 / 64.0  # L3 = L0 / 64 (3 merges)
            return x, dummy_loss, cum_avg_boundaries, cum_boundary_ratio
        else:
            return x

    def forward(self, x, return_loss: bool = False):
        out = self.encode(x, return_loss=return_loss)
        if return_loss:
            feats, boundary_loss, cum_avg_boundaries, cum_ratio = out
        else:
            feats = out

        x = feats.mean(dim=0)                 # (B, C)
        x = self.norm3(x)
        logits = self.head(x)

        if return_loss:
            return logits, boundary_loss, cum_avg_boundaries, cum_ratio
        else:
            return logits

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
        #self.merge = PatchMergeAvg(self.grid0, dim=C)  # L0 -> L1, keep C
        self.merge = TokenPool1DAvg(pool=4, handle_tail="trim")  # L0 -> L1, keep C

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