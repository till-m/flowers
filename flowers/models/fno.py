import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.skip_connections import skip_connection

from einops import rearrange

from flowers.models.utils import create_grid
from flowers.models.utils import FiLM, InstanceNorm


class SpectralBlock(nn.Module):
    """
    Single FNO block with spectral convolution, skip connection, and channel MLP.

    Architecture matches neuralop's FNOBlock:
    - Spectral convolution in Fourier domain
    - Soft-gating skip connection (or linear)
    - Optional FiLM conditioning
    - Channel MLP (optional)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: tuple,
        n_spatial_dims: int = 2,
        use_meta_conditioning: bool = False,
        meta_dim: int = 1,
        film_norm_type: str = 'group',
        num_groups: int = 32,
        activation: str = 'gelu',
        skip_type: str = 'linear',
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_spatial_dims = n_spatial_dims
        self.use_meta_conditioning = use_meta_conditioning

        # Skip connection using neuralop's skip_connection
        self.skip = skip_connection(
            in_features=in_channels,
            out_features=out_channels,
            n_dim=n_spatial_dims,
            skip_type=skip_type,
        )

        # Spectral convolution in Fourier space
        self.spectral_conv = SpectralConv(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=modes,
        )

        # Normalization with optional FiLM conditioning
        if use_meta_conditioning:
            self.norm = FiLM(
                num_channels=out_channels,
                meta_dim=meta_dim,
                norm_type=film_norm_type,
                num_groups=num_groups,
            )
        else:
            if film_norm_type == 'group':
                self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, affine=True)
            elif film_norm_type == 'layer':
                self.norm = nn.LayerNorm(out_channels)
            elif film_norm_type == 'instance':
                self.norm = InstanceNorm()
            else:
                self.norm = nn.Identity()

        # Activation function
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'silu':
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x, meta=None):
        x_spec = self.spectral_conv(x)
        x_skip = self.skip(x)

        out = x_spec + x_skip

        if self.use_meta_conditioning and meta is not None:
            out = self.norm(out, meta)
        else:
            out = self.norm(out)

        out = self.act(out)
        return out


class FNO(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...], # unused, but kept for API consistency
        boundary_condition_types=None,  # unused, but kept for API consistency
        modes: int = 16,
        hidden_channels: int = 128,
        n_layers: int = 4,
        dim_meta: int = 1,
        film_norm_type: str = 'instance',
        num_groups: int = 32,
        activation: str = 'gelu',
        skip_type: str = 'linear',
    ):
        super().__init__()

        self.in_channels = dim_in
        self.out_channels = dim_out
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.n_spatial_dims = n_spatial_dims
        self.use_meta_conditioning = (dim_meta > 0)
        self.dim_meta = dim_meta
        self.spatial_resolution = spatial_resolution

        # Convert modes to tuple if it's an int
        if isinstance(modes, int):
            modes_tuple = tuple([modes] * n_spatial_dims)
        else:
            modes_tuple = tuple(modes)
        self.modes = modes_tuple

        self.input_proj = ChannelMLP(
            in_channels=dim_in + n_spatial_dims + dim_meta,  # Account for concatenated coordinate grid
            out_channels=hidden_channels,
            hidden_channels=hidden_channels * 2,
            n_layers=2,
            n_dim=n_spatial_dims,
            non_linearity=F.gelu if activation == 'gelu' else F.relu,
            dropout=0.0,
        )

        self.blocks = nn.ModuleList([
            SpectralBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                modes=modes_tuple,
                n_spatial_dims=n_spatial_dims,
                use_meta_conditioning=self.use_meta_conditioning,
                meta_dim=dim_meta,
                film_norm_type=film_norm_type,
                num_groups=num_groups,
                activation=activation,
                skip_type=skip_type,
            )
            for _ in range(n_layers)
        ])

        self.output_proj = ChannelMLP(
            in_channels=hidden_channels,
            out_channels=dim_out,
            hidden_channels=hidden_channels * 2,
            n_layers=2,
            n_dim=n_spatial_dims,
            non_linearity=F.gelu if activation == 'gelu' else F.relu,
            dropout=0.0,
        )

        coord_grid = rearrange(
            create_grid(spatial_resolution),
            "... xy -> 1 xy ...",  # '...' are spatial dims
        ).contiguous()
        self.register_buffer("coord_grid", coord_grid)

    def forward(self, x, meta=None):
        batchsize = x.shape[0]

        expand_shape = (batchsize, -1) + (-1,) * self.n_spatial_dims
        grid = self.coord_grid.expand(*expand_shape)

        # If meta provided, broadcast to spatial dimensions and concatenate
        if self.use_meta_conditioning and meta is not None:
            # meta shape: (B, N) -> (B, N, H, W, (D))
            view_shape = (batchsize, self.dim_meta) + (1,) * self.n_spatial_dims
            expand_shape = (-1, -1) + tuple(self.spatial_resolution)
            meta_broadcast = meta.view(*view_shape).expand(*expand_shape)
            grid = torch.cat((grid, meta_broadcast), dim=1)

        x = torch.cat((x, grid), dim=1)

        # Input projection
        x = self.input_proj(x)

        # Pass through spectral blocks
        for block in self.blocks:
            x = block(x, meta)

        # Output projection
        x = self.output_proj(x)

        return x
