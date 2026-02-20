import torch
import torch.nn as nn

def create_grid(spatial_resolution):
    """Create a coordinate grid in [-1, 1] range.
    """
    []
    meshgrid = torch.meshgrid(*[torch.linspace(-1., 1., d_i) for d_i in spatial_resolution])
    grid = torch.stack(meshgrid[::-1], dim=-1) # reverse the meshgrid list
    return grid


def derive_padding_strategies(bc_list, n_spatial_dims):
    """
    Convert boundary condition list to padding strategies.
    """
    if not bc_list:
        return ['zeros' for _ in range(n_spatial_dims)]

    padding = ['periodic' if bc.upper() == 'PERIODIC' else 'zeros' for bc in bc_list]
    
    # If only one element, duplicate it
    if len(padding) == 1:
        padding = [padding[0] for _ in range(n_spatial_dims)]

    return padding


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
            # GroupNorm expects channels_first
            if self.data_format == "channels_last":
                x_for_norm = x.permute(0, -1, *range(1, x.dim() - 1))  # (N, ..., C) -> (N, C, ...)
                x_for_norm = self.norm(x_for_norm)
                x = x_for_norm.permute(0, *range(2, x.dim()), 1)  # (N, C, ...) -> (N, ..., C)
            else:
                x = self.norm(x)
        elif self.norm_type == 'layer':
            # LayerNorm expects channels_last
            if self.data_format == "channels_last":
                x = self.norm(x)
            else:
                x_for_norm = x.permute(0, *range(2, x.dim()), 1)  # (N, C, ...) -> (N, ..., C)
                x_for_norm = self.norm(x_for_norm)
                x = x_for_norm.permute(0, -1, *range(1, x.dim() - 1))  # (N, ..., C) -> (N, C, ...)
        elif self.norm_type == 'identity':
            # Identity norm
            x = self.norm(x)
        else:
            # this should never reach...
            # ...except for the case where I add another norm option and forget to handle it :)
            raise ValueError()

        if meta is None:
            return x

        # Ensure meta has correct shape (N, meta_dim)
        if meta.dim() == 1:
            meta = meta.unsqueeze(-1)  # (N,) -> (N, 1)

        # Map meta features to channel-wise affine parameters
        meta = meta.type_as(x)
        weight = self.weight(meta)  # (N, meta_dim) -> (N, C)
        bias = self.bias(meta)      # (N, meta_dim) -> (N, C)

        # Apply FiLM transformation: scale * normalized(x) + shift
        if self.data_format == "channels_last":
            # weight and bias are (N, C), x is (N, H, W, C)
            # Broadcasting will work correctly
            while weight.dim() < x.dim():
                weight = weight.unsqueeze(1)
                bias = bias.unsqueeze(1)
            return weight * x + bias
        else:
            # For channels_first: x is (N, C, H, W)
            while weight.dim() < x.dim():
                weight = weight.unsqueeze(-1)
                bias = bias.unsqueeze(-1)
            return weight * x + bias


# from the neural operators library
class InstanceNorm(nn.Module):
    """Dimension-agnostic instance normalization layer for neural operators.

    InstanceNorm normalizes each sample in the batch independently, computing
    mean and variance across spatial dimensions for each sample and channel
    separately. This is useful when the statistical properties of each sample
    are distinct and should be treated separately.

    Parameters
    ----------
    **kwargs : dict, optional
        Additional parameters to pass to torch.nn.functional.instance_norm().
        Common parameters include:
        - eps : float, optional
            Small value added to the denominator for numerical stability.
            Default is 1e-5.
        - momentum : float, optional
            Value used for the running_mean and running_var computation.
            Default is 0.1.
        - use_input_stats : bool, optional
            If True, use input statistics. Default is True.
        - weight : torch.Tensor, optional
            Weight tensor for affine transformation. If None, no scaling applied.
        - bias : torch.Tensor, optional
            Bias tensor for affine transformation. If None, no bias applied.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        """Apply instance normalization to the input tensor."""
        size = x.shape
        x = torch.nn.functional.instance_norm(x, **self.kwargs)
        assert x.shape == size
        return x
