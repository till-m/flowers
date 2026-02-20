"""Custom data formatters that extend The Well's formatters to support meta tensors."""

from typing import Dict, Tuple

import torch
from einops import rearrange
from the_well.data.data_formatter import (
    DefaultChannelsFirstFormatter as WellChannelsFirstFormatter,
    DefaultChannelsLastFormatter as WellChannelsLastFormatter,
)


class DefaultChannelsFirstFormatter(WellChannelsFirstFormatter):
    """Extends The Well's DefaultChannelsFirstFormatter to support meta tensors.

    This formatter returns (x, meta) tuple instead of just (x,).
    """

    def process_input(self, data: Dict) -> Tuple:
        """Process input batch and return (inputs, targets) where inputs includes meta.

        Returns:
            inputs: Tuple of (x, meta) where:
                - x: concatenated input fields (B, C, H, W)
                - meta: metadata tensor (B, N) or None
            targets: output fields (B, T, H, W, C)
        """
        x = []
        if "input_fields" in data:
            x.append(rearrange(data["input_fields"], "b t ... c -> b (t c) ..."))
    
        if "constant_fields" in data:
            flat_constants = rearrange(data["constant_fields"], "b ... c -> b c ...")
            x.append(flat_constants)
        x = torch.cat(x, dim=1)

        # Extract meta tensor if present (convert empty tensors to None)
        meta = data.get("meta", None)
        if meta is not None and meta.numel() == 0:
            meta = None

        y = data["output_fields"]
        # Return (x, meta) as inputs, with nan handling
        return (torch.nan_to_num(x), meta), torch.nan_to_num(y)


class DefaultChannelsLastFormatter(WellChannelsLastFormatter):
    """Extends The Well's DefaultChannelsLastFormatter to support meta tensors.

    This formatter returns (x, meta) tuple instead of just (x,).
    """

    def process_input(self, data: Dict) -> Tuple:
        """Process input batch and return (inputs, targets) where inputs includes meta.

        Returns:
            inputs: Tuple of (x, meta) where:
                - x: concatenated input fields (B, H, W, ..., C)
                - meta: metadata tensor (B, N) or None
            targets: output fields (B, T, H, W, ..., C)
        """
        x = data["input_fields"]
        # Flatten time and channels
        x = rearrange(x, "b t ... c -> b ... (t c)")
        if "constant_fields" in data:
            # constant_fields already in channels-last format
            x = torch.cat([x, data["constant_fields"]], dim=-1)

        # Extract meta tensor if present (convert empty tensors to None)
        meta = data.get("meta", None)
        if meta is not None and meta.numel() == 0:
            meta = None

        y = data["output_fields"]
        # Return (x, meta) as inputs, with nan handling
        return (torch.nan_to_num(x), meta), torch.nan_to_num(y)
