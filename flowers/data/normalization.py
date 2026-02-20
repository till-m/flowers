"""Dataset normalization options."""

from typing import Dict, List

import torch


def safe_cat(tensor_list):
    """Helper function to safely concatenate tensors, returning a dummy tensor if empty."""
    if tensor_list:
        return torch.cat(tensor_list, dim=-1)
    else:
        return torch.tensor([1.0])  # Dummy tensor to prevent errors


class ZScoreNormalization:
    @staticmethod
    def _convert_field_first_to_stat_first(field_stats: Dict) -> Dict:
        """Convert field-first format to stat-type-first format.

        Input: {field: {mean: val, std: val, ...}}
        Output: {mean: {field: val}, std: {field: val}, ...}
        """
        stat_first = {}
        for field, stats in field_stats.items():
            for stat_type, value in stats.items():
                if stat_type not in stat_first:
                    stat_first[stat_type] = {}
                stat_first[stat_type][field] = value
        return stat_first

    def __init__(
        self,
        stats: Dict,
        core_field_names: List[str],
        core_constant_field_names: List[str],
        scalar_names: List[str] = None,
        constant_scalar_names: List[str] = None,
        log_transform_scalars: Dict[str, str] = None,
        min_denom: float = 1e-4,
    ):
        """Initialize the Z-Score Normalizer with statistics.

        Args:
            stats: Statistics dictionary in field-first format
            core_field_names: Names of variable fields
            core_constant_field_names: Names of constant fields
            scalar_names: Names of variable scalars
            constant_scalar_names: Names of constant scalars
            log_transform_scalars: Dict mapping scalar name -> normalization type ("log" or "linear")
                For "log" scalars, will use log_mean and log_std from stats
            min_denom: Minimum denominator for normalization

        Stats format (field-first): {field: {mean: val, std: val, mean_delta: val, std_delta: val}}
        For log-transformed scalars: {scalar: {log_mean: val, log_std: val}}
        """
        self.core_field_names = core_field_names
        self.core_constant_field_names = core_constant_field_names
        self.scalar_names = scalar_names or []
        self.constant_scalar_names = constant_scalar_names or []
        self.log_transform_scalars = log_transform_scalars or {}

        # Convert from field-first to stat-type-first format
        stats = self._convert_field_first_to_stat_first(stats)

        required_keys = {"mean", "std", "mean_delta", "std_delta"}
        assert required_keys.issubset(stats.keys()), (
            f"Missing required keys: {required_keys - set(stats.keys())}"
        )

        # All variable and constant field/scalar names
        all_variable_names = core_field_names + self.scalar_names
        all_constant_names = core_constant_field_names + self.constant_scalar_names

        # Store stats for individual fields and scalars
        # For log-transformed scalars, use log_mean and log_std
        self.means = {}
        self.stds = {}
        for field in all_variable_names + all_constant_names:
            if field in self.log_transform_scalars and self.log_transform_scalars[field] == "log":
                # Use log stats for log-transformed scalars
                self.means[field] = torch.as_tensor(stats["log_mean"][field])
                self.stds[field] = torch.clip(torch.as_tensor(stats["log_std"][field]), min=min_denom)
            else:
                # Use regular stats for everything else
                self.means[field] = torch.as_tensor(stats["mean"][field])
                self.stds[field] = torch.clip(torch.as_tensor(stats["std"][field]), min=min_denom)
        self.means_delta = {
            field: torch.as_tensor(stats["mean_delta"][field])
            for field in all_variable_names
        }
        self.stds_delta = {
            field: torch.clip(
                torch.as_tensor(stats["std_delta"].get(field, min_denom)), min=min_denom
            )
            for field in all_variable_names
        }

        # Initialize missing deltas for constant fields and scalars
        self.constant_means_delta = {
            field: torch.full_like(self.means[field], min_denom)
            for field in all_constant_names
        }
        self.constant_stds_delta = {
            field: torch.full_like(self.stds[field], min_denom)
            for field in all_constant_names
        }

        # Precompute flattened stats for each mode
        self._precompute_flattened_stats()

    def _precompute_flattened_stats(self):
        """Precompute mean and std tensors for both variable and constant modes, avoiding empty tensor issues.

        Note: constant mode only includes core_constant_field_names. Constant scalars are passed
        as a separate 'meta' tensor and normalized individually.
        """
        self.flattened_means = {
            "variable": safe_cat(
                [self.means[field].flatten() for field in self.core_field_names]
            ),
            "constant": safe_cat(
                [
                    self.means[field].flatten()
                    for field in self.core_constant_field_names
                ]
            ),
        }
        self.flattened_stds = {
            "variable": safe_cat(
                [self.stds[field].flatten() for field in self.core_field_names]
            ),
            "constant": safe_cat(
                [self.stds[field].flatten() for field in self.core_constant_field_names]
            ),
        }
        self.flattened_means_delta = {
            "variable": safe_cat(
                [self.means_delta[field].flatten() for field in self.core_field_names]
            ),
            "constant": safe_cat(
                [
                    self.constant_means_delta[field].flatten()
                    for field in self.core_constant_field_names
                ]
            ),
        }
        self.flattened_stds_delta = {
            "variable": safe_cat(
                [self.stds_delta[field].flatten() for field in self.core_field_names]
            ),
            "constant": safe_cat(
                [
                    self.constant_stds_delta[field].flatten()
                    for field in self.core_constant_field_names
                ]
            ),
        }

    def normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        """Normalize a single field (field-wise normalization)."""
        assert field in self.means and field in self.stds, (
            f"Field '{field}' not found in statistics."
        )
        return (x - self.means[field]) / self.stds[field]

    def denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        """Denormalize a single field (field-wise denormalization)."""
        assert field in self.means and field in self.stds, (
            f"Field '{field}' not found in statistics."
        )
        return x * self.stds[field] + self.means[field]

    def delta_normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        """Delta normalize a single field (field-wise normalization)."""
        assert field in self.means_delta and field in self.stds_delta, (
            f"Field '{field}' not found in delta statistics."
        )
        return (x - self.means_delta[field]) / self.stds_delta[field]

    def normalize_flattened(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Normalize an input tensor where fields are flattened as channels.

        Args:
            x (torch.Tensor): The input tensor with channels as last dimension.
            mode (str): "variable" for core fields, "constant" for constant fields.
        """
        assert mode in self.flattened_means, (
            f"Invalid mode '{mode}'. Choose from 'variable' or 'constant'."
        )

        mean_values = self.flattened_means[mode].to(x.device)
        std_values = self.flattened_stds[mode].to(x.device)

        assert x.shape[-1] == mean_values.shape[-1], (
            f"Channel mismatch: expected {mean_values.shape[-1]}, got {x.shape[-1]}"
        )
        return (x - mean_values) / std_values

    def denormalize_flattened(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Denormalize an input tensor where fields are flattened as channels."""
        assert mode in self.flattened_means, (
            f"Invalid mode '{mode}'. Choose from 'variable' or 'constant'."
        )

        mean_values = self.flattened_means[mode].to(x.device)
        std_values = self.flattened_stds[mode].to(x.device)

        assert x.shape[-1] == mean_values.shape[-1], (
            f"Channel mismatch: expected {mean_values.shape[-1]}, got {x.shape[-1]}"
        )
        return x * std_values + mean_values

    def delta_normalize_flattened(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Delta normalize an input tensor where fields are flattened as channels."""
        assert mode in self.flattened_means_delta, (
            f"Invalid mode '{mode}'. Choose from 'variable' or 'constant'."
        )

        mean_values = self.flattened_means_delta[mode].to(x.device)
        std_values = self.flattened_stds_delta[mode].to(x.device)

        assert x.shape[-1] == mean_values.shape[-1], (
            f"Channel mismatch: expected {mean_values.shape[-1]}, got {x.shape[-1]}"
        )
        return (x - mean_values) / std_values

    def delta_denormalize_flattened(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Reverses delta normalization for a flattened input tensor."""
        assert mode in self.flattened_means_delta, (
            f"Invalid mode '{mode}'. Choose from 'variable' or 'constant'."
        )

        mean_values = self.flattened_means_delta[mode].to(x.device)
        std_values = self.flattened_stds_delta[mode].to(x.device)

        assert x.shape[-1] == mean_values.shape[-1], (
            f"Channel mismatch: expected {mean_values.shape[-1]}, got {x.shape[-1]}"
        )
        return x * std_values + mean_values


class RMSNormalization:
    @staticmethod
    def _convert_field_first_to_stat_first(field_stats: Dict) -> Dict:
        """Convert field-first format to stat-type-first format.

        Input: {field: {rms: val, rms_delta: val, ...}}
        Output: {rms: {field: val}, rms_delta: {field: val}, ...}
        """
        stat_first = {}
        for field, stats in field_stats.items():
            for stat_type, value in stats.items():
                if stat_type not in stat_first:
                    stat_first[stat_type] = {}
                stat_first[stat_type][field] = value
        return stat_first

    def __init__(
        self,
        stats: Dict,
        core_field_names: List[str],
        core_constant_field_names: List[str],
        scalar_names: List[str] = None,
        constant_scalar_names: List[str] = None,
        log_transform_scalars: Dict[str, str] = None,
        min_denom: float = 1e-4,
    ):
        """Initialize the RMS Normalizer with statistics.

        Args:
            stats: Statistics dictionary in field-first format
            core_field_names: Names of variable fields
            core_constant_field_names: Names of constant fields
            scalar_names: Names of variable scalars
            constant_scalar_names: Names of constant scalars
            log_transform_scalars: Dict mapping scalar name -> normalization type
                (currently unused for RMS normalization)
            min_denom: Minimum denominator for normalization

        Stats format (field-first): {field: {rms: val, rms_delta: val}}
        """
        self.core_field_names = core_field_names
        self.core_constant_field_names = core_constant_field_names
        self.scalar_names = scalar_names or []
        self.constant_scalar_names = constant_scalar_names or []
        self.log_transform_scalars = log_transform_scalars or {}
        # Note: RMS normalization currently doesn't use log-specific stats

        # Convert from field-first to stat-type-first format
        stats = self._convert_field_first_to_stat_first(stats)

        required_keys = {"rms", "rms_delta"}
        assert required_keys.issubset(stats.keys()), (
            f"Missing required keys: {required_keys - set(stats.keys())}"
        )

        # All variable and constant field/scalar names
        all_variable_names = core_field_names + self.scalar_names
        all_constant_names = core_constant_field_names + self.constant_scalar_names

        # Store stats for individual fields and scalars
        self.rmss = {
            field: torch.clip(torch.as_tensor(stats["rms"][field]), min=min_denom)
            for field in all_variable_names + all_constant_names
        }
        self.rmss_delta = {
            field: torch.clip(
                torch.as_tensor(stats["rms_delta"].get(field, min_denom)), min=min_denom
            )
            for field in all_variable_names
        }

        # Initialize missing deltas for constant fields and scalars
        self.constant_rmss_delta = {
            field: torch.full_like(self.rmss[field], min_denom)
            for field in all_constant_names
        }

        # Precompute flattened stats for each mode
        self._precompute_flattened_stats()

    def _precompute_flattened_stats(self):
        """Precompute RMS tensors for 'variable' and 'constant' modes.

        Note: constant mode only includes core_constant_field_names. Constant scalars are passed
        as a separate 'meta' tensor and normalized individually.
        """
        self.flattened_rmss = {
            "variable": safe_cat(
                [self.rmss[field].flatten() for field in self.core_field_names]
            ),
            "constant": safe_cat(
                [self.rmss[field].flatten() for field in self.core_constant_field_names]
            ),
        }
        self.flattened_rmss_delta = {
            "variable": safe_cat(
                [self.rmss_delta[field].flatten() for field in self.core_field_names]
            ),
            "constant": safe_cat(
                [
                    self.constant_rmss_delta[field].flatten()
                    for field in self.core_constant_field_names
                ]
            ),
        }

    def normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        """Normalize a single field (field-wise normalization)."""
        assert field in self.rmss, f"Field '{field}' not found in statistics."
        return x / self.rmss[field]

    def denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        """Denormalize a single field (field-wise denormalization)."""
        assert field in self.rmss, f"Field '{field}' not found in statistics."
        return x * self.rmss[field]

    def delta_normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        """Delta normalize a single field (field-wise normalization)."""
        assert (field in self.rmss_delta) or (field in self.constant_rmss_delta), (
            f"Field '{field}' not found in delta statistics."
        )
        if field in self.rmss_delta:
            return x / self.rmss_delta[field]
        else:
            return x / self.constant_rmss_delta[field]

    def normalize_flattened(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Normalize an input tensor where fields are flattened as channels.

        Args:
            x (torch.Tensor): The input tensor with channels as last dimension.
            mode (str): "variable" for core fields, "constant" for constant fields.
        """
        assert mode in self.flattened_rmss, (
            f"Invalid mode '{mode}'. Choose from 'variable' or 'constant'."
        )

        rms_values = self.flattened_rmss[mode].to(x.device)

        assert x.shape[-1] == rms_values.shape[-1], (
            f"Channel mismatch: expected {rms_values.shape[-1]}, got {x.shape[-1]}"
        )
        return x / rms_values

    def denormalize_flattened(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Denormalize an input tensor where fields are flattened as channels."""
        assert mode in self.flattened_rmss, (
            f"Invalid mode '{mode}'. Choose from 'variable' or 'constant'."
        )

        rms_values = self.flattened_rmss[mode].to(x.device)

        assert x.shape[-1] == rms_values.shape[-1], (
            f"Channel mismatch: expected {rms_values.shape[-1]}, got {x.shape[-1]}"
        )
        return x * rms_values

    def delta_normalize_flattened(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Delta normalize an input tensor where fields are flattened as channels."""
        assert mode in self.flattened_rmss_delta, (
            f"Invalid mode '{mode}'. Choose from 'variable' or 'constant'."
        )

        rms_values = self.flattened_rmss_delta[mode].to(x.device)

        assert x.shape[-1] == rms_values.shape[-1], (
            f"Channel mismatch: expected {rms_values.shape[-1]}, got {x.shape[-1]}"
        )
        return x / rms_values

    def delta_denormalize_flattened(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Reverses delta normalization for a flattened input tensor."""
        assert mode in self.flattened_rmss_delta, (
            f"Invalid mode '{mode}'. Choose from 'variable' or 'constant'."
        )

        rms_values = self.flattened_rmss_delta[mode].to(x.device)

        assert x.shape[-1] == rms_values.shape[-1], (
            f"Channel mismatch: expected {rms_values.shape[-1]}, got {x.shape[-1]}"
        )
        return x * rms_values
