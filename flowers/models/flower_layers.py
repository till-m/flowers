import torch.nn as nn

from einops import rearrange

from flowers.models.utils import custom_grid_sample2d, custom_grid_sample3d, FiLM, create_grid

conv_modules = {
    2: nn.Conv2d,
    3: nn.Conv3d
}
conv_transpose_modules = {
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}

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


class SelfWarp(nn.Module):
    """
    u(x) ⟼ u(x + Δ(x)),

    powered by F.grid_sample
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

        # Cache base grid for grid_sample
        # Note to myself: Don't try to dynamically create this on the forward pass!
        # Even with caching, it's super slow.
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
