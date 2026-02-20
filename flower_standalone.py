"""
This file contains the complete Flower model with all dependencies inlined, for drag-and-drop(-and-import) use in other projects.

Default arguments match configs/models/flower.yaml:
    lifting_dim: 160
    n_levels: 4
    num_heads: 40
    groups: 40
    dropout_rate: 0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# =============================================================================
# Utility Functions
# =============================================================================

def create_grid(spatial_resolution):
    """Create a coordinate grid in [-1, 1] range."""
    meshgrid = torch.meshgrid(*[torch.linspace(-1., 1., d_i) for d_i in spatial_resolution], indexing='ij')
    grid = torch.stack(meshgrid[::-1], dim=-1)  # reverse the meshgrid list
    return grid


def derive_padding_strategies(bc_list, n_spatial_dims):
    """Convert boundary condition list to padding strategies."""
    if not bc_list:
        return ['zeros' for _ in range(n_spatial_dims)]

    padding = ['periodic' if bc.upper() == 'PERIODIC' else 'zeros' for bc in bc_list]

    # If only one element, duplicate it
    if len(padding) == 1:
        padding = [padding[0] for _ in range(n_spatial_dims)]

    return padding


# =============================================================================
# Normalization Layers
# =============================================================================

class InstanceNorm(nn.Module):
    """Dimension-agnostic instance normalization layer for neural operators."""

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        size = x.shape
        x = torch.nn.functional.instance_norm(x, **self.kwargs)
        assert x.shape == size
        return x


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer with flexible normalization."""

    def __init__(self, num_channels, meta_dim=1, norm_type='layer', num_groups=32, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.num_channels = num_channels
        self.meta_dim = meta_dim
        self.norm_type = norm_type.lower()
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (num_channels,)

        # Set up normalization layer based on type
        if self.norm_type == 'group':
            self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=False)
        elif self.norm_type == 'layer':
            self.norm = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=False)
        elif self.norm_type == 'instance':
            self.norm = InstanceNorm()
        elif self.norm_type == 'identity':
            self.norm = nn.Identity()
        else:
            raise ValueError(f"norm_type must be 'group', 'layer', 'instance', or 'identity', got {norm_type}")

        # Map from meta_dim to channel affine parameters
        self.weight = nn.Linear(meta_dim, num_channels)
        self.bias = nn.Linear(meta_dim, num_channels)

    def forward(self, x, meta=None):
        if self.norm_type in ['group', 'instance']:
            if self.data_format == "channels_last":
                x_for_norm = x.permute(0, -1, *range(1, x.dim() - 1))
                x_for_norm = self.norm(x_for_norm)
                x = x_for_norm.permute(0, *range(2, x.dim()), 1)
            else:
                x = self.norm(x)
        elif self.norm_type == 'layer':
            if self.data_format == "channels_last":
                x = self.norm(x)
            else:
                x_for_norm = x.permute(0, *range(2, x.dim()), 1)
                x_for_norm = self.norm(x_for_norm)
                x = x_for_norm.permute(0, -1, *range(1, x.dim() - 1))
        elif self.norm_type == 'identity':
            x = self.norm(x)
        else:
            raise ValueError()

        if meta is None:
            return x

        if meta.dim() == 1:
            meta = meta.unsqueeze(-1)

        meta = meta.type_as(x)
        weight = self.weight(meta)
        bias = self.bias(meta)

        if self.data_format == "channels_last":
            while weight.dim() < x.dim():
                weight = weight.unsqueeze(1)
                bias = bias.unsqueeze(1)
            return weight * x + bias
        else:
            while weight.dim() < x.dim():
                weight = weight.unsqueeze(-1)
                bias = bias.unsqueeze(-1)
            return weight * x + bias


# =============================================================================
# Custom Grid Sampling with Periodic Boundary Support
# =============================================================================

@torch.compile
def custom_grid_sample2d(
    input, grid, mode="bilinear", padding_modes=("zeros", "zeros"), align_corners=False
):
    """
    Custom grid_sample with per-dimension periodic boundary conditions for 2D.

    Args:
        input: Input tensor of shape (N, C, H, W)
        grid: Grid tensor of shape (N, H, W, 2) with normalized coordinates in [-1, 1]
        mode: Interpolation mode ('bilinear', 'nearest')
        padding_modes: Tuple (padding_h, padding_w) where each is one of
                      {'zeros', 'border', 'reflection', 'periodic'}.
        align_corners: Whether to align corners

    Returns:
        Sampled tensor of shape (N, C, H, W)
    """
    padding_h, padding_w = padding_modes

    if padding_w.lower() != "periodic" and padding_h.lower() != "periodic":
        if padding_w != padding_h:
            msg = "When using different paddings for x and y dimensions, one should be `periodic`."
            raise ValueError(msg)
        return F.grid_sample(
            input, grid, mode=mode, padding_mode=padding_w, align_corners=align_corners
        )

    periodic_w = padding_w == "periodic"
    periodic_h = padding_h == "periodic"
    final_padding_w = "border" if periodic_w else padding_w
    final_padding_h = "border" if periodic_h else padding_h

    N, C, H, W = input.shape

    if periodic_h and periodic_w:
        top_row = input[:, :, 0:1, :]
        left_col = input[:, :, :, 0:1]
        top_left_corner = input[:, :, 0:1, 0:1]
        padded_h = torch.cat([input, top_row], dim=2)
        left_col_extended = torch.cat([left_col, top_left_corner], dim=2)
        padded_input = torch.cat([padded_h, left_col_extended], dim=3)
        H_padded = H + 1
        W_padded = W + 1
    elif periodic_h:
        top_row = input[:, :, 0:1, :]
        padded_input = torch.cat([input, top_row], dim=2)
        H_padded = H + 1
        W_padded = W
    elif periodic_w:
        left_col = input[:, :, :, 0:1]
        padded_input = torch.cat([input, left_col], dim=3)
        H_padded = H
        W_padded = W + 1

    gx, gy = grid[..., 0], grid[..., 1]

    if align_corners:
        px = (gx + 1) * (W - 1) / 2
        py = (gy + 1) * (H - 1) / 2

        if periodic_w:
            px_wrapped = px % W
        else:
            px_wrapped = px

        if periodic_h:
            py_wrapped = py % H
        else:
            py_wrapped = py

        if periodic_w:
            gx_new = 2 * px_wrapped / W - 1
        else:
            gx_new = 2 * px_wrapped / (W_padded - 1) - 1 if W_padded > 1 else gx

        if periodic_h:
            gy_new = 2 * py_wrapped / H - 1
        else:
            gy_new = 2 * py_wrapped / (H_padded - 1) - 1 if H_padded > 1 else gy
    else:
        px = ((gx + 1) * W - 1) / 2
        py = ((gy + 1) * H - 1) / 2

        if periodic_w:
            px_wrapped = px % W
        else:
            px_wrapped = px

        if periodic_h:
            py_wrapped = py % H
        else:
            py_wrapped = py

        if periodic_w:
            gx_new = 2 * (px_wrapped + 0.5) / (W + 1) - 1
        else:
            gx_new = 2 * (px_wrapped + 0.5) / W_padded - 1 if W_padded > 0 else gx

        if periodic_h:
            gy_new = 2 * (py_wrapped + 0.5) / (H + 1) - 1
        else:
            gy_new = 2 * (py_wrapped + 0.5) / H_padded - 1 if H_padded > 0 else gy

    grid_new = torch.stack([gx_new, gy_new], dim=-1)

    if periodic_w and periodic_h:
        final_padding = "border"
    elif periodic_w:
        final_padding = final_padding_h
    elif periodic_h:
        final_padding = final_padding_w

    return F.grid_sample(
        padded_input,
        grid_new,
        mode=mode,
        padding_mode=final_padding,
        align_corners=align_corners,
    )


@torch.compile
def custom_grid_sample3d(input,
                          grid,
                          mode="bilinear",
                          padding_modes=("periodic", "periodic", "periodic"),
                          align_corners=True):
    """
    3D grid_sample with support for per-dimension periodic boundary conditions.

    Args:
        input: (N, C, D, H, W)
        grid:  (N, D_out, H_out, W_out, 3), normalized coords in [-1, 1]
        mode: 'bilinear' (trilinear) or 'nearest'
        padding_modes: 3-tuple with entries in {'zeros', 'border', 'reflection', 'periodic'}.
        align_corners: as in torch.nn.functional.grid_sample
    """
    padding_d, padding_h, padding_w = padding_modes

    periodic_d = (padding_d == "periodic")
    periodic_h = (padding_h == "periodic")
    periodic_w = (padding_w == "periodic")

    if not (periodic_d or periodic_h or periodic_w):
        if not (padding_d == padding_h == padding_w):
            raise ValueError(
                "When using different paddings for D, H, W and none is 'periodic', "
                "they must all be equal."
            )
        return F.grid_sample(
            input,
            grid,
            mode=mode,
            padding_mode=padding_d,
            align_corners=align_corners,
        )

    non_periodic_paddings = []
    if not periodic_d:
        non_periodic_paddings.append(padding_d)
    if not periodic_h:
        non_periodic_paddings.append(padding_h)
    if not periodic_w:
        non_periodic_paddings.append(padding_w)

    if len(non_periodic_paddings) == 0:
        final_padding = "border"
    else:
        if len(set(non_periodic_paddings)) > 1:
            raise ValueError(
                "Non-periodic dimensions must use the same padding_mode, "
                f"got: {set(non_periodic_paddings)}"
            )
        final_padding = non_periodic_paddings[0]

    N, C, D, H, W = input.shape

    padded_input = input
    D_padded, H_padded, W_padded = D, H, W

    if periodic_d:
        front_slice = padded_input[:, :, 0:1, :, :]
        padded_input = torch.cat([padded_input, front_slice], dim=2)
        D_padded += 1

    if periodic_h:
        top_slice = padded_input[:, :, :, 0:1, :]
        padded_input = torch.cat([padded_input, top_slice], dim=3)
        H_padded += 1

    if periodic_w:
        left_slice = padded_input[:, :, :, :, 0:1]
        padded_input = torch.cat([padded_input, left_slice], dim=4)
        W_padded += 1

    gx = grid[..., 0]
    gy = grid[..., 1]
    gz = grid[..., 2]

    if align_corners:
        px = (gx + 1.0) * (W - 1.0) / 2.0
        py = (gy + 1.0) * (H - 1.0) / 2.0
        pz = (gz + 1.0) * (D - 1.0) / 2.0

        px_wrapped = px % W if periodic_w else px
        py_wrapped = py % H if periodic_h else py
        pz_wrapped = pz % D if periodic_d else pz

        if periodic_w:
            gx_new = 2.0 * px_wrapped / W - 1.0
        else:
            gx_new = 2.0 * px_wrapped / (W_padded - 1.0) - 1.0 if W_padded > 1 else gx

        if periodic_h:
            gy_new = 2.0 * py_wrapped / H - 1.0
        else:
            gy_new = 2.0 * py_wrapped / (H_padded - 1.0) - 1.0 if H_padded > 1 else gy

        if periodic_d:
            gz_new = 2.0 * pz_wrapped / D - 1.0
        else:
            gz_new = 2.0 * pz_wrapped / (D_padded - 1.0) - 1.0 if D_padded > 1 else gz

    else:
        px = ((gx + 1.0) * W - 1.0) / 2.0
        py = ((gy + 1.0) * H - 1.0) / 2.0
        pz = ((gz + 1.0) * D - 1.0) / 2.0

        px_wrapped = px % W if periodic_w else px
        py_wrapped = py % H if periodic_h else py
        pz_wrapped = pz % D if periodic_d else pz

        if periodic_w:
            gx_new = 2.0 * (px_wrapped + 0.5) / (W + 1.0) - 1.0
        else:
            gx_new = 2.0 * (px_wrapped + 0.5) / W_padded - 1.0 if W_padded > 0 else gx

        if periodic_h:
            gy_new = 2.0 * (py_wrapped + 0.5) / (H + 1.0) - 1.0
        else:
            gy_new = 2.0 * (py_wrapped + 0.5) / H_padded - 1.0 if H_padded > 0 else gy

        if periodic_d:
            gz_new = 2.0 * (pz_wrapped + 0.5) / (D + 1.0) - 1.0
        else:
            gz_new = 2.0 * (pz_wrapped + 0.5) / D_padded - 1.0 if D_padded > 0 else gz

    grid_new = torch.stack([gx_new, gy_new, gz_new], dim=-1)

    return F.grid_sample(
        padded_input,
        grid_new,
        mode=mode,
        padding_mode=final_padding,
        align_corners=align_corners,
    )


# =============================================================================
# Module Dictionaries
# =============================================================================

conv_modules = {2: nn.Conv2d, 3: nn.Conv3d}
conv_transpose_modules = {2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}

flow_fold_heads = {
    2: 'B (heads dir) H W -> (B heads) H W dir',
    3: 'B (heads dir) D H W -> (B heads) D H W dir',
}

value_fold_heads = {
    2: 'B (heads C_i) H W -> (B heads) C_i H W',
    3: 'B (heads C_i) D H W -> (B heads) C_i D H W'
}

value_unfold_heads = {
    2: '(B heads) C_i H W -> B (heads C_i) H W',
    3: '(B heads) C_i D H W -> B (heads C_i) D H W'
}

grid_samples = {
    2: custom_grid_sample2d,
    3: custom_grid_sample3d
}


# =============================================================================
# Flower Layers
# =============================================================================

class SelfWarp(nn.Module):
    """
    Spatial warping operator: u(x) -> u(x + delta(x))

    Powered by F.grid_sample with support for periodic boundaries.
    """

    def __init__(self, in_channels, out_channels, spatial_resolution, num_heads=32, padding_modes=["zeros", "zeros"], meta_dim=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_spatial_dims = len(spatial_resolution)
        self.spatial_resolution = spatial_resolution

        self.padding_modes = padding_modes
        self.num_heads = num_heads

        self.flow_head = nn.Sequential(
            conv_modules[self.n_spatial_dims](in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            conv_modules[self.n_spatial_dims](out_channels, self.n_spatial_dims * self.num_heads, kernel_size=1)
        )
        self.value_head = conv_modules[self.n_spatial_dims](in_channels, out_channels, kernel_size=1)

        base_grid = create_grid(self.spatial_resolution)
        self.register_buffer("base_grid", base_grid)

    def flow(self, u, return_value=False):
        flow = self.flow_head(u)
        if return_value:
            value = self.value_head(u)
            return flow, value
        return flow

    def forward(self, u, meta=None):
        # u: (B, C, H, W)
        flow = self.flow_head(u)
        value = self.value_head(u)

        # === Vectorized grid_sample computation ===
        # To make the grid_sample efficient, we can vectorize it by folding the different heads
        # into the batch dimension. However, this makes the code a bit messy since we need to
        # reshape the tensors a bunch.

        # Convert to (B, ..., dir) format for grid_sample
        flow = rearrange(flow, flow_fold_heads[self.n_spatial_dims], dir=self.n_spatial_dims, heads=self.num_heads)

        # Create warped grid for each (batch x head)
        base_grid_batch = self.base_grid.unsqueeze(0) # (B heads) H W dir
        grid = base_grid_batch + flow  # Add displacement

        # Apply spatial warping using grid_sample
        value = rearrange(value, value_fold_heads[self.n_spatial_dims], heads=self.num_heads)
        u_warp = grid_samples[self.n_spatial_dims](
            value,
            grid,
            mode='bilinear',
            padding_modes=self.padding_modes,
            align_corners=True
        )
        # unfold the heads
        u_warp = rearrange(u_warp, value_unfold_heads[self.n_spatial_dims], heads=self.num_heads)
        # === End vectorized computation ===
        return u_warp


class FlowerBlock(nn.Module):
    """Core building block of Flower: SelfWarp + residual + normalization."""

    def __init__(self,
            in_channels,
            out_channels,
            spatial_resolution,
            num_heads,
            padding_modes,
            use_meta_conditioning,
            dropout_rate=0.0,
            groups=32,
            meta_dim=1,
            film_norm_type='group'
        ):
        super().__init__()
        self.self_warp = SelfWarp(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_resolution=spatial_resolution,
            num_heads=num_heads,
            padding_modes=padding_modes,
            meta_dim=meta_dim
        )
        self.n_spatial_dims = len(spatial_resolution)
        self.spatial_resolution = spatial_resolution
        self.id = conv_modules[self.n_spatial_dims](in_channels, out_channels, 1)
        self.use_meta_conditioning = use_meta_conditioning

        if use_meta_conditioning:
            self.norm = FiLM(
                num_channels=out_channels,
                meta_dim=meta_dim,
                norm_type=film_norm_type,
                num_groups=groups
            )
        else:
            self.norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, affine=True)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x, meta=None):
        id_ = self.id(x)
        out = self.self_warp(x, meta) + id_

        if self.use_meta_conditioning and meta is not None:
            out = self.norm(out, meta)
        else:
            out = self.norm(out)

        return self.dropout(self.act(out))


# =============================================================================
# U-Net Blocks
# =============================================================================

class DownsampleBlock(nn.Module):
    """Encoder block: FlowerBlock + strided convolution downsampling."""

    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_resolution,
        num_heads,
        padding_modes,
        use_meta_conditioning=False,
        meta_dim=1,
        groups=32,
        dropout_rate=0.0
    ):
        super().__init__()
        self.n_spatial_dims = len(spatial_resolution)
        self.spatial_resolution = spatial_resolution

        self.shift_layer = FlowerBlock(
            in_channels,
            in_channels,
            spatial_resolution=spatial_resolution,
            num_heads=num_heads,
            padding_modes=padding_modes,
            use_meta_conditioning=use_meta_conditioning,
            dropout_rate=dropout_rate,
            meta_dim=meta_dim,
            groups=groups,
        )
        self.downsample = conv_modules[self.n_spatial_dims](
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, meta=None):
        x_down = self.shift_layer(x, meta)
        x_down = self.act(self.downsample(x_down))
        return x, x_down


class UpsampleBlock(nn.Module):
    """Decoder block: FlowerBlock + transposed convolution upsampling."""

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        spatial_resolution,
        num_heads,
        padding_modes,
        use_meta_conditioning=False,
        meta_dim=1,
        groups=32,
        dropout_rate=0.0
    ):
        super().__init__()
        self.n_spatial_dims = len(spatial_resolution)
        self.spatial_resolution = spatial_resolution

        self.fio = FlowerBlock(
            in_channels,
            out_channels - skip_channels,
            spatial_resolution=spatial_resolution,
            num_heads=num_heads,
            padding_modes=padding_modes,
            use_meta_conditioning=use_meta_conditioning,
            dropout_rate=dropout_rate,
            meta_dim=meta_dim,
            groups=groups,
        )

        self.upsample = conv_transpose_modules[self.n_spatial_dims](
            out_channels - skip_channels, out_channels - skip_channels, kernel_size=2, stride=2
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, meta=None):
        x = self.fio(x, meta)
        x = self.act(self.upsample(x))
        return x


# =============================================================================
# Main Flower Model
# =============================================================================

class Flower(nn.Module):
    """
    Flower: A U-Net style neural operator with spatial warping layers.

    Default arguments match configs/models/flower.yaml:
        lifting_dim: 160
        n_levels: 4
        num_heads: 40
        groups: 40
        dropout_rate: 0.0

    Args:
        dim_in: Number of input channels (set from dataset)
        dim_out: Number of output channels (set from dataset)
        n_spatial_dims: Number of spatial dimensions (2 or 3)
        spatial_resolution: Tuple of spatial dimensions, e.g. (128, 128)
        lifting_dim: Hidden dimension after lifting layer
        n_levels: Number of U-Net levels (encoder/decoder pairs)
        num_heads: Number of attention heads in spatial warping
        boundary_condition_types:
            List of boundary conditions per dimension. Only "PERIODIC" is treated differently
        dim_meta: Dimension of metadata for conditioning (0 = no conditioning)
        groups: Number of groups for GroupNorm
        dropout_rate: Dropout probability
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
        lifting_dim: int = 160,
        n_levels: int = 4,
        num_heads: int = 40,
        boundary_condition_types: list[str] = ["PERIODIC"],
        dim_meta: int = 0,
        groups: int = 40,
        dropout_rate: float = 0.0
    ):
        super().__init__()

        self.input_dim = dim_in
        self.output_dim = dim_out
        self.spatial_resolution = spatial_resolution
        self.lifting_dim = lifting_dim
        self.n_levels = n_levels
        self.dim_meta = dim_meta
        self.use_meta_conditioning = (dim_meta > 0)
        self.num_heads = num_heads
        self.boundary_condition_types = boundary_condition_types
        self.padding_type = derive_padding_strategies(
            self.boundary_condition_types, n_spatial_dims=n_spatial_dims
        )
        self.n_spatial_dims = n_spatial_dims
        self.spatial_resolution = spatial_resolution
        self.groups = groups

        min_divisor = 2 ** (n_levels - 1)
        for i, d_i in enumerate(spatial_resolution):
            if d_i % min_divisor != 0:
                if d_i == 66:  # post_neutron_star_merger
                    spatial_resolution[i] = 64
                    continue
                else:
                    raise ValueError(
                        f"Dimension {i} must be divisible by 2^(n_levels-1) = {min_divisor}, "
                        f"or n_levels={n_levels}, but is {d_i}."
                    )

        coord_dim = self.n_spatial_dims
        if self.use_meta_conditioning:
            coord_dim += dim_meta

        self.lift = conv_modules[self.n_spatial_dims](
            dim_in + coord_dim, lifting_dim, kernel_size=1
        )

        # Calculate channel dimensions for each level
        channel_multipliers = [2.0**i for i in range(n_levels)]
        self.encoder_channels = [
            int(lifting_dim * mult) for mult in channel_multipliers
        ]

        # Encoder path (downsampling)
        self.encoder_blocks = nn.ModuleList()
        current_spatial_resolution = self.spatial_resolution

        for i in range(n_levels - 1):
            in_ch = self.encoder_channels[i]
            out_ch = self.encoder_channels[i + 1]
            self.encoder_blocks.append(
                DownsampleBlock(
                    in_ch,
                    out_ch,
                    current_spatial_resolution,
                    num_heads,
                    self.padding_type,
                    self.use_meta_conditioning,
                    meta_dim=dim_meta,
                    groups=self.groups,
                    dropout_rate=dropout_rate
                )
            )
            current_spatial_resolution = [
                d_i // 2 for d_i in current_spatial_resolution
            ]

        # Bottleneck at the deepest level
        bottleneck_channels = self.encoder_channels[-1]

        self.bottleneck = FlowerBlock(
            bottleneck_channels,
            bottleneck_channels,
            spatial_resolution=current_spatial_resolution,
            num_heads=num_heads,
            padding_modes=self.padding_type,
            use_meta_conditioning=self.use_meta_conditioning,
            meta_dim=dim_meta,
            groups=groups,
            dropout_rate=dropout_rate
        )

        # Decoder path (upsampling)
        self.decoder_blocks = nn.ModuleList()

        decoder_in_ch = bottleneck_channels

        for i in range(n_levels - 1, 0, -1):
            skip_ch = self.encoder_channels[i - 1]
            out_ch = skip_ch * 2

            self.decoder_blocks.append(
                UpsampleBlock(
                    decoder_in_ch,
                    skip_ch,
                    out_ch,
                    current_spatial_resolution,
                    num_heads,
                    self.padding_type,
                    self.use_meta_conditioning,
                    meta_dim=dim_meta,
                    groups=self.groups,
                    dropout_rate=dropout_rate
                )
            )

            decoder_in_ch = out_ch
            current_spatial_resolution = [2 * d_i for d_i in current_spatial_resolution]

        # Final projection
        self.project = nn.Sequential(
            conv_modules[self.n_spatial_dims](
                out_ch, lifting_dim, kernel_size=1
            ),
            nn.ReLU(inplace=True),
            conv_modules[self.n_spatial_dims](lifting_dim, dim_out, kernel_size=1),
        )
        self.dropout = nn.Identity() if dropout_rate == 0.0 else nn.Dropout(p=dropout_rate)

        # Cache positional encoding grid
        coord_grid = rearrange(
            create_grid(spatial_resolution),
            "... xy -> 1 xy ...",
        ).contiguous()
        self.register_buffer("coord_grid", coord_grid)

    def forward(self, x, meta=None):
        batchsize = x.shape[0]

        # Handle 3D data with last dimension 66 (i.e., neutron star merger)
        needs_66_interpolation = self.n_spatial_dims == 3 and x.shape[-1] == 66
        if needs_66_interpolation:
            x = F.interpolate(
                x,
                size=(x.shape[2], x.shape[3], 64),
                mode="trilinear",
                align_corners=False,
            )

        expand_shape = (batchsize, -1) + (-1,) * self.n_spatial_dims
        grid = self.coord_grid.expand(*expand_shape)

        if self.use_meta_conditioning and meta is not None:
            view_shape = (batchsize, self.dim_meta) + (1,) * self.n_spatial_dims
            expand_shape = (-1, -1) + tuple(self.spatial_resolution)
            meta_broadcast = meta.view(*view_shape).expand(*expand_shape)
            grid = torch.cat((grid, meta_broadcast), dim=1)

        x = torch.cat((x, grid), dim=1)

        x = self.lift(x)

        # Encoder path with skip connections
        skip_connections = []

        for encoder_block in self.encoder_blocks:
            skip, x = encoder_block(x, meta)
            skip_connections.append(skip)

        x = self.bottleneck(x, meta)

        # Decoder path
        for decoder_block, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = decoder_block(x, meta)
            x = torch.cat([x, skip], dim=1)

        x = self.dropout(x)
        prediction = self.project(x)

        # Interpolate back to 66 if we interpolated down
        if needs_66_interpolation:
            prediction = F.interpolate(
                prediction,
                size=(prediction.shape[2], prediction.shape[3], 66),
                mode="trilinear",
                align_corners=False,
            )

        return prediction


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: 2D problem with 4 input channels and 4 output channels
    model = Flower(
        dim_in=4,
        dim_out=4,
        n_spatial_dims=2,
        spatial_resolution=(128, 128),
        # Default args from flower.yaml:
        lifting_dim=160,
        n_levels=4,
        num_heads=40,
        groups=40,
        dropout_rate=0.0,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Flower model created with {num_params:,} parameters")

    # Test forward pass
    x = torch.randn(2, 4, 128, 128)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
