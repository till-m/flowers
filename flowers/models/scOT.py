"""
This file contains scOT, taken from POSEIDON, and modified.

The difference between poseidon-scOT.py and scOT.py is that here we refactored the code to (a) accept non-square images and (b) not depend on
HuggingFace config objects. This mirrors the set-up of the other models.
-----------------------------------------------------------

A lot of this file is taken from the transformers library and changed to our purposes. Huggingface Transformers is licensed under
Apache 2.0 License, see trainer.py for details.

We follow https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/swinv2/configuration_swinv2.py
and https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/swinv2/modeling_swinv2.py#L1129

The class ConvNeXtBlock is taken from the facebookresearch/ConvNeXt repository and is licensed under the MIT License,

MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from transformers import (
    PretrainedConfig,
)
from transformers.models.swinv2.modeling_swinv2 import (
    Swinv2EncoderOutput,
    Swinv2Attention,
    Swinv2DropPath,
    Swinv2Intermediate,
    Swinv2Output,
    window_reverse,
    window_partition,
)
from transformers.utils import ModelOutput
from dataclasses import dataclass
import torch
from torch import nn
from typing import Optional, Union, Tuple, List
import math
import collections


@dataclass
class ScOTOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class ScOTConfig(PretrainedConfig):
    """https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/swinv2/configuration_swinv2.py"""

    model_type = "swinv2"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        image_size=224,
        patch_size=4,
        num_channels=3,
        num_out_channels=1,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        skip_connections=[True, True, True],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        p=1,  # for loss: 1 for l1, 2 for l2
        channel_slice_list_normalized_loss=None,  # if None will fall back to absolute loss otherwise normalized loss with split channels
        residual_model="convnext",  # "convnext" or "resnet"
        dim_meta=0,
        use_meta_conditioning=False,
        learn_residual=False,  # learn the residual for time-dependent problems
        spatial_resolution=None,  # Alternative to image_size for non-square images
        **kwargs,
    ):
        super().__init__(**kwargs)

        if spatial_resolution is not None:
            # Use spatial_resolution if provided (matches fionet signature)
            # Check if it's a sequence (handles list, tuple, ListConfig, etc.)
            if isinstance(spatial_resolution, collections.abc.Sequence) and not isinstance(spatial_resolution, str):
                if len(spatial_resolution) == 2:
                    # Force convert to plain Python ints
                    self.image_size = tuple(int(x) if hasattr(x, '__int__') else x for x in spatial_resolution)
                else:
                    raise ValueError(f"spatial_resolution must have length 2, got {len(spatial_resolution)}")
            else:
                # It's a scalar - duplicate it
                self.image_size = (int(spatial_resolution), int(spatial_resolution))
        elif isinstance(image_size, collections.abc.Sequence) and not isinstance(image_size, str):
            self.image_size = tuple(int(x) if hasattr(x, '__int__') else x for x in image_size)
        else:
            self.image_size = (int(image_size), int(image_size))

        # Handle patch_size - convert ListConfig or other sequences to tuple of ints
        if isinstance(patch_size, collections.abc.Sequence) and not isinstance(patch_size, str):
            self.patch_size = tuple(int(x) if hasattr(x, '__int__') else x for x in patch_size)
        else:
            self.patch_size = int(patch_size) if hasattr(patch_size, '__int__') else patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = list(depths) if not isinstance(depths, list) else depths
        self.num_layers = len(depths)
        self.num_heads = list(num_heads) if not isinstance(num_heads, list) else num_heads
        self.skip_connections = list(skip_connections) if not isinstance(skip_connections, list) else skip_connections
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_absolute_embeddings = use_absolute_embeddings
        self.dim_meta = dim_meta
        self.use_meta_conditioning = use_meta_conditioning or (dim_meta > 0)

        self.use_conditioning = self.use_meta_conditioning
        self.learn_residual = learn_residual if self.use_meta_conditioning else False
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        # we set the hidden_size attribute in order to make Swinv2 work with VisionEncoderDecoderModel
        # this indicates the channel dimension after the last stage of the model
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))
        self.pretrained_window_sizes = (0, 0, 0, 0)
        self.num_out_channels = num_out_channels
        self.p = p
        self.channel_slice_list_normalized_loss = channel_slice_list_normalized_loss
        self.residual_model = residual_model


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, meta):
        return super().forward(x)


class ConditionalLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, meta_dim=1):
        super().__init__()
        self.eps = eps
        self.meta_dim = meta_dim
        self.weight = nn.Linear(meta_dim, dim)
        self.bias = nn.Linear(meta_dim, dim)

    def forward(self, x, meta):
        mean = x.mean(dim=-1, keepdim=True)
        var = (x**2).mean(dim=-1, keepdim=True) - mean**2
        x = (x - mean) / (var + self.eps).sqrt()
        # meta should be (batch_size, meta_dim)
        if meta.dim() == 1:
            meta = meta.unsqueeze(-1)
        meta = meta.type_as(x)
        weight = self.weight(meta).unsqueeze(1)
        bias = self.bias(meta).unsqueeze(1)
        if x.dim() == 4:
            weight = weight.unsqueeze(1)
            bias = bias.unsqueeze(1)
        return weight * x + bias


class ConvNeXtBlock(nn.Module):
    r"""Taken from: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, config, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        if config.use_conditioning:
            self.norm = ConditionalLayerNorm(dim, eps=config.layer_norm_eps, meta_dim=config.dim_meta)
        else:
            self.norm = LayerNorm(dim, eps=config.layer_norm_eps)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.weight = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )  # was gamma before
        self.drop_path = Swinv2DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, meta, input_dimensions):
        batch_size, sequence_length, hidden_size = x.shape

        height, width = input_dimensions

        input = x
        x = x.reshape(batch_size, height, width, hidden_size)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x, meta)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.weight is not None:
            x = self.weight * x
        x = x.reshape(batch_size, sequence_length, hidden_size)

        x = input + self.drop_path(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        kernel_size = 3
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=pad)
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x, meta, input_dimensions=None):
        batch_size, sequence_length, hidden_size = x.shape

        height, width = input_dimensions

        input = x
        x = x.reshape(batch_size, height, width, hidden_size)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, sequence_length, hidden_size)
        x = x + input
        return x


class ScOTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim

        if isinstance(image_size, collections.abc.Iterable):
            image_size = tuple(int(x) if hasattr(x, '__int__') else x for x in image_size)
        else:
            image_size = (int(image_size), int(image_size))

        if isinstance(patch_size, collections.abc.Iterable):
            patch_size = tuple(int(x) if hasattr(x, '__int__') else x for x in patch_size)
        else:
            patch_size = (int(patch_size), int(patch_size))

        num_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0]
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )

        self.projection = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def maybe_pad(self, pixel_values, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    def forward(
        self, pixel_values: Optional[torch.FloatTensor]
    ) -> Tuple[torch.Tensor, Tuple[int]]:
        _, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # pad the input to be divisible by self.patch_size, if needed
        pixel_values = self.maybe_pad(pixel_values, height, width)
        embeddings = self.projection(pixel_values)
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings, output_dimensions


class ScOTEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token=False):
        super().__init__()

        self.patch_embeddings = ScOTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            if use_mask_token
            else None
        )

        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, num_patches, config.embed_dim)
            )
        else:
            self.position_embeddings = None

        if config.use_conditioning:
            self.norm = ConditionalLayerNorm(config.embed_dim, meta_dim=config.dim_meta)
        else:
            self.norm = LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor],
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        meta: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings, meta)
        batch_size, seq_len, _ = embeddings.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings, output_dimensions


class ScOTLayer(nn.Module):
    def __init__(
        self,
        config,
        dim,
        input_resolution,
        num_heads,
        drop_path=0.0,
        shift_size=0,
        pretrained_window_size=0,
    ):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.shift_size = shift_size
        self.window_size = config.window_size if isinstance(config.window_size, int) else tuple(config.window_size)
        self.input_resolution = input_resolution
        self.set_shift_and_window_size(input_resolution)
        self.attention = Swinv2Attention(
            config=config,
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            pretrained_window_size=(
                tuple(pretrained_window_size)
                if isinstance(pretrained_window_size, collections.abc.Iterable)
                else (pretrained_window_size, pretrained_window_size)
            ),
        )
        if config.use_conditioning:
            self.layernorm_before = ConditionalLayerNorm(dim, eps=config.layer_norm_eps, meta_dim=config.dim_meta)
            self.layernorm_after = ConditionalLayerNorm(dim, eps=config.layer_norm_eps, meta_dim=config.dim_meta)
        else:
            self.layernorm_before = LayerNorm(dim, eps=config.layer_norm_eps)
            self.layernorm_after = LayerNorm(dim, eps=config.layer_norm_eps)
        self.drop_path = Swinv2DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.intermediate = Swinv2Intermediate(config, dim)
        self.output = Swinv2Output(config, dim)
        
        # Cache for attention masks
        self.attn_mask_cache = {}
        # Cache for padding calculations
        self.pad_cache = {}

    def set_shift_and_window_size(self, input_resolution):
        target_window_size = (
            self.window_size
            if isinstance(self.window_size, collections.abc.Iterable)
            else (self.window_size, self.window_size)
        )
        target_shift_size = (
            self.shift_size
            if isinstance(self.shift_size, collections.abc.Iterable)
            else (self.shift_size, self.shift_size)
        )
        window_dim = (
            input_resolution[0].item()
            if torch.is_tensor(input_resolution[0])
            else input_resolution[0]
        )
        self.window_size = (
            window_dim if window_dim <= target_window_size[0] else target_window_size[0]
        )
        self.shift_size = (
            0
            if input_resolution
            <= (
                self.window_size
                if isinstance(self.window_size, collections.abc.Iterable)
                else (self.window_size, self.window_size)
            )
            else target_shift_size[0]
        )

    def get_attn_mask(self, height, width, dtype):
        # Use cached attention mask when possible
        cache_key = (height, width, self.shift_size, self.window_size, dtype)
        if cache_key in self.attn_mask_cache:
            return self.attn_mask_cache[cache_key]
            
        if self.shift_size > 0:
            # calculate attention mask for shifted window multihead self attention
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
            
        # Cache the result
        self.attn_mask_cache[cache_key] = attn_mask
        return attn_mask

    def maybe_pad(self, hidden_states, height, width):
        # Use cached padding calculations when possible
        cache_key = (height, width, self.window_size)
        if cache_key in self.pad_cache:
            pad_values = self.pad_cache[cache_key]
            if pad_values[3] > 0 or pad_values[5] > 0:
                hidden_states = nn.functional.pad(hidden_states, pad_values)
            return hidden_states, pad_values
            
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        
        # Cache the pad values
        self.pad_cache[cache_key] = pad_values
        
        if pad_right > 0 or pad_bottom > 0:
            hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        meta: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not always_partition:
            self.set_shift_and_window_size(input_dimensions)
            
        height, width = input_dimensions
        batch_size, seq_len, channels = hidden_states.size()
        shortcut = hidden_states

        # Reshape and pad in one step if needed
        hidden_states = hidden_states.view(batch_size, height, width, channels)
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)
        _, height_pad, width_pad, _ = hidden_states.shape
        
        # Only apply cyclic shift if needed
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(
                hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_hidden_states = hidden_states

        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(
            -1, self.window_size * self.window_size, channels
        )
        
        # Get attention mask (cached when possible)
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)
            
        attention_outputs = self.attention(
            hidden_states_windows,
            attn_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs[0]
        
        # Reconstruct feature map
        attention_windows = attention_output.view(
            -1, self.window_size, self.window_size, channels
        )
        shifted_windows = window_reverse(
            attention_windows, self.window_size, height_pad, width_pad
        )
        
        # Reverse cyclic shift if needed
        if self.shift_size > 0:
            attention_windows = torch.roll(
                shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            attention_windows = shifted_windows
            
        # Handle padding if necessary
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()
            
        attention_windows = attention_windows.view(batch_size, height * width, channels)

        hidden_states = shortcut + self.drop_path(self.layernorm_before(attention_windows, meta))

        residual = hidden_states
        layer_output = self.output(self.intermediate(hidden_states))
        layer_output = residual + self.drop_path(self.layernorm_after(layer_output, meta))
        
        layer_outputs = (
            (layer_output, attention_outputs[1])
            if output_attentions
            else (layer_output,)
        )
        return layer_outputs


class ScOTPatchRecovery(nn.Module):
    """https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py"""

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_out_channels, hidden_size = (
            config.num_out_channels,
            config.embed_dim,  # if not config.skip_connections[0] else 2 * config.embed_dim,
        )

        if isinstance(image_size, collections.abc.Iterable):
            image_size = tuple(int(x) if hasattr(x, '__int__') else x for x in image_size)
        else:
            image_size = (int(image_size), int(image_size))

        if isinstance(patch_size, collections.abc.Iterable):
            patch_size = tuple(int(x) if hasattr(x, '__int__') else x for x in patch_size)
        else:
            patch_size = (int(patch_size), int(patch_size))
        num_patches = (image_size[0] // patch_size[0]) * (
            image_size[1] // patch_size[1]
        )
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_out_channels = num_out_channels
        self.grid_size = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )

        self.projection = nn.ConvTranspose2d(
            in_channels=hidden_size,
            out_channels=num_out_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )
        # the following is not done in Pangu
        self.mixup = nn.Conv2d(
            num_out_channels,
            num_out_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )

    def maybe_crop(self, pixel_values, height, width):
        if pixel_values.shape[2] > height:
            pixel_values = pixel_values[:, :, :height, :]
        if pixel_values.shape[3] > width:
            pixel_values = pixel_values[:, :, :, :width]
        return pixel_values

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], hidden_states.shape[1], *self.grid_size
        )

        output = self.projection(hidden_states)
        output = self.maybe_crop(output, self.image_size[0], self.image_size[1])
        return self.mixup(output)


class ScOTPatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(
        self, input_resolution: Tuple[int], dim: int, norm_layer: nn.Module = LayerNorm, config=None
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

        if config is not None and config.use_conditioning:
            self.norm = norm_layer(2 * dim, meta_dim=config.dim_meta)
        else:
            self.norm = norm_layer(2 * dim)

    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = nn.functional.pad(input_feature, pad_values)

        return input_feature

    def forward(
        self,
        input_feature: torch.Tensor,
        input_dimensions: Tuple[int, int],
        meta: torch.Tensor,
    ) -> torch.Tensor:
        height, width = input_dimensions
        # `dim` is height * width
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.view(batch_size, height, width, num_channels)
        # pad input to be disible by width and height, if needed
        input_feature = self.maybe_pad(input_feature, height, width)
        # [batch_size, height/2, width/2, num_channels]
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        # [batch_size, height/2 * width/2, 4*num_channels]
        input_feature = torch.cat(
            [input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1
        )
        input_feature = input_feature.view(
            batch_size, -1, 4 * num_channels
        )  # [batch_size, height/2 * width/2, 4*C]

        input_feature = self.reduction(input_feature)
        input_feature = self.norm(input_feature, meta)

        return input_feature


class ScOTPatchUnmerging(nn.Module):
    def __init__(
        self,
        input_resolution: Tuple[int],
        dim: int,
        norm_layer: nn.Module = LayerNorm,
        config=None,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.upsample = nn.Linear(dim, 2 * dim, bias=False)
        self.mixup = nn.Linear(dim // 2, dim // 2, bias=False)

        if config is not None and config.use_conditioning:
            self.norm = norm_layer(dim // 2, meta_dim=config.dim_meta)
        else:
            self.norm = norm_layer(dim // 2)

    def maybe_crop(self, input_feature, height, width):
        height_in, width_in = input_feature.shape[1], input_feature.shape[2]
        if height_in > height:
            input_feature = input_feature[:, :height, :, :]
        if width_in > width:
            input_feature = input_feature[:, :, :width, :]
        return input_feature

    def forward(
        self,
        input_feature: torch.Tensor,
        output_dimensions: Tuple[int, int],
        meta: torch.Tensor,
        input_dimensions: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        output_height, output_width = output_dimensions
        batch_size, seq_len, hidden_size = input_feature.shape

        input_height, input_width = input_dimensions

        input_feature = self.upsample(input_feature)
        input_feature = input_feature.reshape(
            batch_size, input_height, input_width, 2, 2, hidden_size // 2
        )
        input_feature = input_feature.permute(0, 1, 3, 2, 4, 5)
        input_feature = input_feature.reshape(
            batch_size, 2 * input_height, 2 * input_width, hidden_size // 2
        )

        input_feature = self.maybe_crop(input_feature, output_height, output_width)
        input_feature = input_feature.reshape(batch_size, -1, hidden_size // 2)

        input_feature = self.norm(input_feature, meta)
        return self.mixup(input_feature)


class ScOTEncodeStage(nn.Module):
    def __init__(
        self,
        config,
        dim,
        input_resolution,
        depth,
        num_heads,
        drop_path,
        downsample,
        pretrained_window_size=0,
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        window_size = (
            tuple(config.window_size)
            if isinstance(config.window_size, collections.abc.Iterable)
            else (config.window_size, config.window_size)
        )
        self.blocks = nn.ModuleList(
            [
                ScOTLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=(
                        [0, 0]
                        if (i % 2 == 0)
                        else [window_size[0] // 2, window_size[1] // 2]
                    ),
                    drop_path=drop_path[i],
                    pretrained_window_size=pretrained_window_size,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            if config.use_conditioning:
                layer_norm = ConditionalLayerNorm
            else:
                layer_norm = LayerNorm
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=layer_norm, config=config
            )
        else:
            self.downsample = None

        self.pointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        meta: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        height, width = input_dimensions

        inputs = hidden_states

        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                meta,
                layer_head_mask,
                output_attentions,
                always_partition,
            )

            hidden_states = layer_outputs[0]

        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(
                hidden_states_before_downsampling + inputs, input_dimensions, meta
            )
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (
            hidden_states,
            hidden_states_before_downsampling,
            output_dimensions,
        )

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


class ScOTDecodeStage(nn.Module):
    def __init__(
        self,
        config,
        dim,
        input_resolution,
        depth,
        num_heads,
        drop_path,
        upsample,
        upsampled_size,
        pretrained_window_size=0,
    ):
        super().__init__()
        self.config = config
        self.dim = dim
        window_size = (
            tuple(config.window_size)
            if isinstance(config.window_size, collections.abc.Iterable)
            else (config.window_size, config.window_size)
        )
        self.blocks = nn.ModuleList(
            [
                ScOTLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=(
                        [0, 0]
                        if (i % 2 == 0)
                        else [window_size[0] // 2, window_size[1] // 2]
                    ),
                    drop_path=drop_path[depth - 1 - i],  # TODO: reverse...
                    pretrained_window_size=pretrained_window_size,
                )
                for i in reversed(range(depth))  # TODO: reverse here?
            ]
        )

        if upsample is not None:
            if config.use_conditioning:
                layer_norm = ConditionalLayerNorm
            else:
                layer_norm = LayerNorm
            self.upsample = upsample(input_resolution, dim=dim, norm_layer=layer_norm, config=config)
            self.upsampled_size = upsampled_size
        else:
            self.upsample = None

        self.pointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        meta: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        height, width = input_dimensions

        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                meta,
                layer_head_mask,
                output_attentions,
                always_partition,
            )

            hidden_states = layer_outputs[0]

        hidden_states_before_upsampling = hidden_states
        if self.upsample is not None:
            height_upsampled, width_upsampled = self.upsampled_size
            output_dimensions = (height, width, height_upsampled, width_upsampled)
            hidden_states = self.upsample(
                hidden_states_before_upsampling,
                (height_upsampled, width_upsampled),
                meta,
                input_dimensions=(height, width),
            )
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (
            hidden_states,
            hidden_states_before_upsampling,
            output_dimensions,
        )

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


class ScOTEncoder(nn.Module):
    """
    This is just a Swinv2Encoder with changed dpr.
    We just have to change the drop path rate since we also have a decoder by default.
    """

    def __init__(self, config, grid_size, pretrained_window_sizes=(0, 0, 0, 0)):
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        if self.config.pretrained_window_sizes is not None:
            pretrained_window_sizes = config.pretrained_window_sizes
        drop_rates_encode_decode = torch.linspace(
            0, config.drop_path_rate, 2 * sum(config.depths)
        )
        dpr = [
            x.item()
            for x in drop_rates_encode_decode[: drop_rates_encode_decode.shape[0] // 2]
        ]
        self.layers = nn.ModuleList(
            [
                ScOTEncodeStage(
                    config=config,
                    dim=int(config.embed_dim * 2**i_layer),
                    input_resolution=(
                        grid_size[0] // (2**i_layer),
                        grid_size[1] // (2**i_layer),
                    ),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    drop_path=dpr[
                        sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])
                    ],
                    downsample=(
                        ScOTPatchMerging if (i_layer < self.num_layers - 1) else None
                    ),
                    pretrained_window_size=pretrained_window_sizes[i_layer],
                )
                for i_layer in range(self.num_layers)
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        meta: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, Swinv2EncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            # rearrange b (h w) c -> b c h w
            reshaped_hidden_state = hidden_states.view(
                batch_size, *input_dimensions, hidden_size
            )
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    input_dimensions,
                    meta,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    input_dimensions,
                    meta,
                    layer_head_mask,
                    output_attentions,
                    always_partition,
                )

            hidden_states = layer_outputs[0]
            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                # rearrange b (h w) c -> b c h w
                # here we use the original (not downsampled) height and width
                reshaped_hidden_state = hidden_states_before_downsampling.view(
                    batch_size,
                    *(output_dimensions[0], output_dimensions[1]),
                    hidden_size,
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange b (h w) c -> b c h w
                reshaped_hidden_state = hidden_states.view(
                    batch_size, *input_dimensions, hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[3:]

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return Swinv2EncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


class ScOTDecoder(nn.Module):
    """Here we do reverse encoder."""

    def __init__(self, config, grid_size, pretrained_window_sizes=(0, 0, 0, 0)):
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        if self.config.pretrained_window_sizes is not None:
            pretrained_window_sizes = config.pretrained_window_sizes
        drop_rates_encode_decode = torch.linspace(
            0, config.drop_path_rate, 2 * sum(config.depths)
        )
        dpr = [
            x.item()
            for x in drop_rates_encode_decode[drop_rates_encode_decode.shape[0] // 2 :]
        ]
        self.layers = nn.ModuleList(
            [
                ScOTDecodeStage(
                    config=config,
                    dim=int(config.embed_dim * 2**i_layer),
                    input_resolution=(
                        grid_size[0] // (2**i_layer),
                        grid_size[1] // (2**i_layer),
                    ),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    drop_path=dpr[
                        sum(config.depths[i_layer + 1 :]) : sum(config.depths[i_layer:])
                    ],
                    upsample=ScOTPatchUnmerging if i_layer > 0 else None,
                    upsampled_size=(
                        grid_size[0] // (2 ** (i_layer - 1)),
                        grid_size[1] // (2 ** (i_layer - 1)),
                    ),
                    pretrained_window_size=pretrained_window_sizes[i_layer],
                )
                for i_layer in reversed(range(self.num_layers))
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        skip_states: List[torch.FloatTensor],
        meta: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_upsampling: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, Swinv2EncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            # rearrange b (h w) c -> b c h w
            reshaped_hidden_state = hidden_states.view(
                batch_size, *input_dimensions, hidden_size
            )
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            if i != 0 and skip_states[len(skip_states) - i] is not None:
                # residual connection
                hidden_states = hidden_states + skip_states[len(skip_states) - i]
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    input_dimensions,
                    meta,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    input_dimensions,
                    meta,
                    layer_head_mask,
                    output_attentions,
                    always_partition,
                )

            hidden_states = layer_outputs[0]
            hidden_states_before_upsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            if output_hidden_states and output_hidden_states_before_upsampling:
                batch_size, _, hidden_size = hidden_states_before_upsampling.shape
                # rearrange b (h w) c -> b c h w
                # here we use the original (not downsampled) height and width
                reshaped_hidden_state = hidden_states_before_upsampling.view(
                    batch_size,
                    *(output_dimensions[0], output_dimensions[1]),
                    hidden_size,
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_upsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_upsampling:
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange b (h w) c -> b c h w
                reshaped_hidden_state = hidden_states.view(
                    batch_size, *input_dimensions, hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                all_self_attentions += layer_outputs[3:]

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return Swinv2EncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )


class ScOT(nn.Module):
    """Inspired by https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/swinv2/modeling_swinv2.py#L1129

    Refactored to not use a HuggingFace config object.
    """

    def __init__(
        self,
        dim_in=3,
        dim_out=1,
        spatial_resolution=(128, 128),
        n_spatial_dims=2,
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        skip_connections=[1, 1, 1],  # Number of residual blocks per stage
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        layer_norm_eps=1e-5,
        residual_model="convnext",
        dim_meta=0,
        learn_residual=False,
        use_mask_token=False,
        **kwargs
    ):
        super().__init__()
        use_meta_conditioning = (dim_meta > 0)

        if n_spatial_dims != 2:
            raise NotImplementedError

        # Handle spatial_resolution
        if isinstance(spatial_resolution, (list, tuple)) and len(spatial_resolution) == 2:
            image_size = tuple(spatial_resolution)
        else:
            image_size = (spatial_resolution, spatial_resolution)

        # Create a config object internally
        config = ScOTConfig(
            image_size=image_size[0] if image_size[0] == image_size[1] else image_size,
            patch_size=patch_size,
            num_channels=dim_in,
            num_out_channels=dim_out,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            skip_connections=skip_connections,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            drop_path_rate=drop_path_rate,
            hidden_act=hidden_act,
            use_absolute_embeddings=use_absolute_embeddings,
            layer_norm_eps=layer_norm_eps,
            residual_model=residual_model,
            dim_meta=dim_meta,
            use_meta_conditioning=use_meta_conditioning,
            learn_residual=learn_residual,
            spatial_resolution=spatial_resolution,
        )

        self.config = config
        self.num_layers_encoder = len(config.depths)
        self.num_layers_decoder = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers_encoder - 1))

        self.embeddings = ScOTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ScOTEncoder(config, self.embeddings.patch_grid)
        self.decoder = ScOTDecoder(config, self.embeddings.patch_grid)
        self.patch_recovery = ScOTPatchRecovery(config)

        if config.residual_model == "convnext":
            res_model = ConvNeXtBlock
        elif config.residual_model == "resnet":
            res_model = ResNetBlock
        else:
            raise ValueError("residual_model must be 'convnext' or 'resnet'")

        self.residual_blocks = nn.ModuleList(
            [
                (
                    nn.ModuleList(
                        [
                            res_model(config, config.embed_dim * 2**i)
                            for _ in range(depth)
                        ]
                    )
                    if depth > 0
                    else nn.ModuleList([nn.Identity()])
                )
                for i, depth in enumerate(config.skip_connections)
            ]
        )

    def _downsample(self, image, target_size):
        image_size = image.shape[-2]
        freqs = torch.fft.fftfreq(image_size, d=1 / image_size)
        sel = torch.logical_and(freqs >= -target_size / 2, freqs <= target_size / 2 - 1)
        image_hat = torch.fft.fft2(image, norm="forward")
        image_hat = image_hat[:, :, sel, :][:, :, :, sel]
        image = torch.fft.ifft2(image_hat, norm="forward").real
        return image

    def _upsample(self, image, target_size):
        # https://stackoverflow.com/questions/71143279/upsampling-images-in-frequency-domain-using-pytorch
        image_size = image.shape[-2]
        image_hat = torch.fft.fft2(image, norm="forward")
        image_hat = torch.fft.fftshift(image_hat)
        pad_size = (target_size - image_size) // 2
        real = nn.functional.pad(
            image_hat.real, (pad_size, pad_size, pad_size, pad_size), value=0.0
        )
        imag = nn.functional.pad(
            image_hat.imag, (pad_size, pad_size, pad_size, pad_size), value=0.0
        )
        image_hat = torch.fft.ifftshift(torch.complex(real, imag))
        image = torch.fft.ifft2(image_hat, norm="forward").real
        return image

    def forward(
        self,
        x: Optional[torch.FloatTensor] = None,
        meta: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # Simplified forward for 2D non-rectangular inputs
        pixel_values = x
        if pixel_values is None:
            raise ValueError("x cannot be None")

        # Get head masks (using None as default)
        head_mask = None
        head_mask_encoder = None
        head_mask_decoder = None

        # No resizing - work with input as-is
        embedding_output, input_dimensions = self.embeddings(
            pixel_values, bool_masked_pos=None, meta=meta
        )

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            meta,
            head_mask=head_mask_encoder,
            output_attentions=False,
            output_hidden_states=True,
            output_hidden_states_before_downsampling=True,
            return_dict=True,
        )

        skip_states = list(encoder_outputs.hidden_states[1:])

        # Apply residual blocks to skip connections (not the bottleneck)
        for i in range(len(self.residual_blocks)):
            # Calculate spatial dimensions for this skip connection level
            # skip_states[i] contains the state BEFORE downsampling in stage i
            # So skip_states[0] is at the patch grid level (after 0 downsamples)
            # skip_states[1] is after 1 downsample, etc.
            height_at_level = input_dimensions[0] // (2 ** i)
            width_at_level = input_dimensions[1] // (2 ** i)
            dims_at_level = (height_at_level, width_at_level)

            for block in self.residual_blocks[i]:
                if isinstance(block, nn.Identity):
                    skip_states[i] = block(skip_states[i])
                else:
                    skip_states[i] = block(skip_states[i], meta, input_dimensions=dims_at_level)

        # Use the patch grid from embeddings to get proper dimensions
        height_patches = self.embeddings.patch_grid[0] // (2 ** (len(self.config.depths) - 1))
        width_patches = self.embeddings.patch_grid[1] // (2 ** (len(self.config.depths) - 1))

        decoder_output = self.decoder(
            skip_states[-1],
            (height_patches, width_patches),
            skip_states=skip_states[:-1],
            meta=meta,
            head_mask=head_mask_decoder,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

        sequence_output = decoder_output[0]
        prediction = self.patch_recovery(sequence_output)

        # Optional: learn residual for time-dependent problems
        if self.config.learn_residual:
            if self.config.num_channels > self.config.num_out_channels:
                pixel_values = pixel_values[:, 0 : self.config.num_out_channels]
            prediction = prediction + pixel_values

        return prediction
