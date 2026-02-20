"""Calculate normalization statistics for The Well datasets."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml
from tqdm import tqdm


def round_to_n_sig_figs(x, n=5):
    if x == 0:
        return 0.0
    return float(f"{x:.{n-1}e}")


def format_float_scientific(x):
    return f"{x:.4E}"


yaml.add_representer(float, lambda d, v: d.represent_scalar('tag:yaml.org,2002:float', format_float_scientific(v)))


def welford_update(count, mean, M2, sum_sq, new_values):
    """Welford's algorithm for online mean/variance/RMS computation."""
    new_values_flat = new_values.ravel()
    n_new = len(new_values_flat)
    if n_new == 0:
        return count, mean, M2, sum_sq

    sum_sq += np.sum(new_values_flat ** 2)
    new_mean = np.mean(new_values_flat)
    new_M2 = np.sum((new_values_flat - new_mean) ** 2)

    total_count = count + n_new
    delta = new_mean - mean
    combined_mean = mean + delta * n_new / total_count
    combined_M2 = M2 + new_M2 + delta ** 2 * count * n_new / total_count

    return total_count, combined_mean, combined_M2, sum_sq


def log_welford_update(count, mean, M2, has_nonpositive, new_values):
    """Welford's algorithm for log-transformed values."""
    new_values_flat = new_values.ravel()
    if np.any(new_values_flat <= 0) or has_nonpositive:
        return count, mean, M2, True

    log_values = np.log(new_values_flat)
    n_new = len(log_values)
    new_mean = np.mean(log_values)
    new_M2 = np.sum((log_values - new_mean) ** 2)

    total_count = count + n_new
    delta = new_mean - mean
    combined_mean = mean + delta * n_new / total_count
    combined_M2 = M2 + new_M2 + delta ** 2 * count * n_new / total_count

    return total_count, combined_mean, combined_M2, False


def init_stats_dict(with_deltas=False):
    """Initialize statistics accumulator dictionary."""
    d = {"count": 0, "mean": 0.0, "M2": 0.0, "sum_sq": 0.0,
         "log_count": 0, "log_mean": 0.0, "log_M2": 0.0, "log_has_nonpositive": False}
    if with_deltas:
        d.update({"count_delta": 0, "mean_delta": 0.0, "M2_delta": 0.0, "sum_sq_delta": 0.0})
    return d


def update_component_stats(stats, data, with_deltas=False):
    """Update statistics for a single component."""
    stats["count"], stats["mean"], stats["M2"], stats["sum_sq"] = welford_update(
        stats["count"], stats["mean"], stats["M2"], stats["sum_sq"], data)
    stats["log_count"], stats["log_mean"], stats["log_M2"], stats["log_has_nonpositive"] = log_welford_update(
        stats["log_count"], stats["log_mean"], stats["log_M2"], stats["log_has_nonpositive"], data)
    if with_deltas and data.ndim >= 2:
        deltas = np.diff(data, axis=1)
        stats["count_delta"], stats["mean_delta"], stats["M2_delta"], stats["sum_sq_delta"] = welford_update(
            stats["count_delta"], stats["mean_delta"], stats["M2_delta"], stats["sum_sq_delta"], deltas)


def compute_stats_from_hdf5(dataset_path: Path, split_name: str, n_spatial_dims: int, verbose: bool = True):
    """Compute normalization statistics by reading HDF5 files directly."""
    split_dir = dataset_path / "data" / split_name
    hdf5_files = sorted(list(split_dir.glob("*.hdf5")) + list(split_dir.glob("*.h5")))
    if len(hdf5_files) == 0:
        raise ValueError(f"No HDF5 files found in {split_dir}")

    if verbose:
        print(f"\nFound {len(hdf5_files)} HDF5 files in {split_dir}")

    # Validate all files before processing
    if verbose:
        print("\nValidating HDF5 files...")
    corrupted_files = []
    for file_path in tqdm(hdf5_files, desc="Validating files", disable=not verbose):
        try:
            with h5py.File(file_path, "r") as f:
                # Just opening is enough to trigger truncation errors
                pass
        except (OSError, IOError) as e:
            corrupted_files.append((file_path, str(e)))

    if corrupted_files:
        print(f"\n{'='*80}")
        print(f"ERROR: Found {len(corrupted_files)} corrupted/truncated HDF5 files:")
        print(f"{'='*80}")
        for file_path, error in corrupted_files:
            print(f"  {file_path.name}")
            print(f"    Error: {error}")
        print(f"{'='*80}")
        raise ValueError(f"Cannot proceed: {len(corrupted_files)} corrupted files found. Please fix/redownload these files.")

    field_stats = {}
    constant_field_stats = {}
    constant_scalar_stats = {}

    for file_path in tqdm(hdf5_files, desc=f"Processing {split_name}", disable=not verbose):
        with h5py.File(file_path, "r") as f:
            field_groups = []
            if "fields" in f:
                field_groups.append(f["fields"])
            for key in f.keys():
                if key.startswith("t") and key.endswith("_fields"):
                    field_groups.append(f[key])

            for field_group in field_groups:
                field_names = list(field_group.attrs.get("field_names", []))

                for field_name in field_names:
                    if field_name not in field_group:
                        continue
                    field_dataset = field_group[field_name]
                    is_time_varying = field_dataset.attrs.get("time_varying", True)
                    data = field_dataset[:]

                    if is_time_varying:
                        expected_ndim = 2 + n_spatial_dims
                        tensor_order_attr = field_dataset.attrs.get("tensor_order", None)

                        if tensor_order_attr == 1:
                            tensor_shape = (data.shape[-1],)
                        elif tensor_order_attr == 2:
                            tensor_shape = data.shape[-2:]
                        elif data.ndim > expected_ndim:
                            tensor_shape = data.shape[expected_ndim:]
                        else:
                            tensor_shape = None

                        if tensor_shape:
                            if field_name not in field_stats:
                                field_stats[field_name] = {"tensor_shape": tensor_shape, "components": {}}

                            if len(tensor_shape) == 1:
                                for i in range(tensor_shape[0]):
                                    if i not in field_stats[field_name]["components"]:
                                        field_stats[field_name]["components"][i] = init_stats_dict(with_deltas=True)
                                    update_component_stats(field_stats[field_name]["components"][i], data[..., i], with_deltas=True)
                            elif len(tensor_shape) == 2:
                                for i in range(tensor_shape[0]):
                                    for j in range(tensor_shape[1]):
                                        comp_key = (i, j)
                                        if comp_key not in field_stats[field_name]["components"]:
                                            field_stats[field_name]["components"][comp_key] = init_stats_dict(with_deltas=True)
                                        update_component_stats(field_stats[field_name]["components"][comp_key], data[..., i, j], with_deltas=True)
                        else:
                            if field_name not in field_stats:
                                field_stats[field_name] = init_stats_dict(with_deltas=True)
                            update_component_stats(field_stats[field_name], data, with_deltas=True)
                    else:
                        if field_name not in constant_field_stats:
                            constant_field_stats[field_name] = init_stats_dict(with_deltas=False)
                        update_component_stats(constant_field_stats[field_name], data, with_deltas=False)

            if "scalars" in f:
                scalar_names = list(f["scalars"].attrs.get("field_names", []))
                for scalar_name in scalar_names:
                    if scalar_name not in f["scalars"]:
                        continue
                    scalar_dataset = f["scalars"][scalar_name]
                    is_time_varying = scalar_dataset.attrs.get("time_varying", False)
                    data = np.array([scalar_dataset[()]]) if scalar_dataset.shape == () else scalar_dataset[:]

                    if is_time_varying:
                        if scalar_name not in field_stats:
                            field_stats[scalar_name] = init_stats_dict(with_deltas=True)
                        update_component_stats(field_stats[scalar_name], data, with_deltas=True)
                    else:
                        if scalar_name not in constant_scalar_stats:
                            constant_scalar_stats[scalar_name] = init_stats_dict(with_deltas=False)
                        update_component_stats(constant_scalar_stats[scalar_name], data, with_deltas=False)

    def finalize_stats(stats_dict, has_deltas=False):
        result = {}
        for name, data in stats_dict.items():
            if "components" in data:
                result[name] = {"tensor_shape": data["tensor_shape"], "components": {}}
                for comp_key, comp_data in data["components"].items():
                    if comp_data["count"] > 0:
                        result[name]["components"][comp_key] = {
                            "mean": round_to_n_sig_figs(comp_data["mean"]),
                            "std": round_to_n_sig_figs(np.sqrt(comp_data["M2"] / comp_data["count"])),
                            "rms": round_to_n_sig_figs(np.sqrt(comp_data["sum_sq"] / comp_data["count"])),
                        }
                        if comp_data["log_has_nonpositive"]:
                            result[name]["components"][comp_key]["log_mean"] = float('nan')
                            result[name]["components"][comp_key]["log_std"] = float('nan')
                        else:
                            result[name]["components"][comp_key]["log_mean"] = round_to_n_sig_figs(comp_data["log_mean"])
                            result[name]["components"][comp_key]["log_std"] = round_to_n_sig_figs(np.sqrt(comp_data["log_M2"] / comp_data["log_count"]))
                        if has_deltas and comp_data["count_delta"] > 0:
                            result[name]["components"][comp_key]["mean_delta"] = round_to_n_sig_figs(comp_data["mean_delta"])
                            result[name]["components"][comp_key]["std_delta"] = round_to_n_sig_figs(np.sqrt(comp_data["M2_delta"] / comp_data["count_delta"]))
                            result[name]["components"][comp_key]["rms_delta"] = round_to_n_sig_figs(np.sqrt(comp_data["sum_sq_delta"] / comp_data["count_delta"]))
            else:
                if data["count"] > 0:
                    result[name] = {
                        "mean": round_to_n_sig_figs(data["mean"]),
                        "std": round_to_n_sig_figs(np.sqrt(data["M2"] / data["count"])),
                        "rms": round_to_n_sig_figs(np.sqrt(data["sum_sq"] / data["count"])),
                    }
                    if data["log_has_nonpositive"]:
                        result[name]["log_mean"] = float('nan')
                        result[name]["log_std"] = float('nan')
                    else:
                        result[name]["log_mean"] = round_to_n_sig_figs(data["log_mean"])
                        result[name]["log_std"] = round_to_n_sig_figs(np.sqrt(data["log_M2"] / data["log_count"]))
                    if has_deltas and data["count_delta"] > 0:
                        result[name]["mean_delta"] = round_to_n_sig_figs(data["mean_delta"])
                        result[name]["std_delta"] = round_to_n_sig_figs(np.sqrt(data["M2_delta"] / data["count_delta"]))
                        result[name]["rms_delta"] = round_to_n_sig_figs(np.sqrt(data["sum_sq_delta"] / data["count_delta"]))
        return result

    field_results = finalize_stats(field_stats, has_deltas=True)
    constant_field_results = finalize_stats(constant_field_stats, has_deltas=False)
    constant_scalar_results = finalize_stats(constant_scalar_stats, has_deltas=False)

    if verbose:
        print(f"\nComputed stats for {len(field_results)} fields, {len(constant_field_results)} constant fields, {len(constant_scalar_results)} scalars")

    return {"fields": field_results, "constant_fields": constant_field_results, "constant_scalars": constant_scalar_results}


def format_stats_for_yaml(stats):
    """Format statistics dictionary grouped by variable."""
    output = {}
    all_fields = {**stats.get("fields", {}), **stats.get("constant_fields", {}), **stats.get("constant_scalars", {})}

    for field_name, field_data in all_fields.items():
        output[field_name] = {}
        if "components" in field_data:
            tensor_shape = field_data["tensor_shape"]
            stat_types = set()
            for comp_data in field_data["components"].values():
                stat_types.update(comp_data.keys())

            if len(tensor_shape) == 1:
                n_components = tensor_shape[0]
                for stat_type in stat_types:
                    stat_values = [field_data["components"][i][stat_type] for i in range(n_components)
                                   if i in field_data["components"] and stat_type in field_data["components"][i]]
                    if stat_values:
                        output[field_name][stat_type] = stat_values
            elif len(tensor_shape) == 2:
                n_i, n_j = tensor_shape
                for stat_type in stat_types:
                    stat_matrix = []
                    for i in range(n_i):
                        row = [field_data["components"][(i, j)][stat_type] for j in range(n_j)
                               if (i, j) in field_data["components"] and stat_type in field_data["components"][(i, j)]]
                        if row:
                            stat_matrix.append(row)
                    if stat_matrix:
                        output[field_name][stat_type] = stat_matrix
        else:
            output[field_name] = field_data
    return output


def main():
    parser = argparse.ArgumentParser(description="Calculate normalization statistics for The Well datasets")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ndims", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--base-path", type=str, default="data/the-well/datasets")
    parser.add_argument("--splits", type=str, nargs="+", default=["train"])
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.base_path) / args.dataset
    output_path = f"{args.dataset}_stats.yaml"

    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    print(f"{'='*80}\nCalculating stats for: {args.dataset} ({args.ndims}D)\n{'='*80}")

    all_split_stats = []
    for split in args.splits:
        print(f"\n{'='*80}\nProcessing split: {split}\n{'='*80}")
        split_stats = compute_stats_from_hdf5(dataset_path, split_name=split, n_spatial_dims=args.ndims, verbose=True)
        all_split_stats.append(split_stats)

    combined_stats = all_split_stats[0]
    formatted_stats = format_stats_for_yaml(combined_stats)

    print(f"\n{'='*80}\nWriting to: {output_path}\n{'='*80}")
    with open(output_path, "w") as f:
        yaml.dump(formatted_stats, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Done")

    if args.compare:
        existing_stats_path = dataset_path / "stats.yaml"
        if existing_stats_path.exists():
            print(f"\n{'='*80}\nCOMPARISON:\n{'='*80}")
            with open(existing_stats_path, "r") as f:
                existing_stats = yaml.safe_load(f)

            for field_name in formatted_stats.keys():
                print(f"\n{field_name}:")
                for stat_type in ["mean", "std", "rms", "mean_delta", "std_delta", "rms_delta"]:
                    if stat_type not in formatted_stats[field_name]:
                        continue
                    if stat_type in existing_stats and field_name in existing_stats[stat_type]:
                        old_val = existing_stats[stat_type][field_name]
                        new_val = formatted_stats[field_name][stat_type]

                        if isinstance(old_val, list) and isinstance(new_val, list):
                            print(f"  {stat_type}:")
                            if old_val and isinstance(old_val[0], list):
                                for i, (o_row, n_row) in enumerate(zip(old_val, new_val)):
                                    for j, (o, n) in enumerate(zip(o_row, n_row)):
                                        diff_pct = abs(o - n) / (abs(o) + 1e-10) * 100
                                        print(f"    [{i},{j}] Old: {o:12.4e}  New: {n:12.4e}  Diff: {diff_pct:6.2f}%")
                            else:
                                for i, (o, n) in enumerate(zip(old_val, new_val)):
                                    diff_pct = abs(o - n) / (abs(o) + 1e-10) * 100
                                    print(f"    [{i}] Old: {o:12.4e}  New: {n:12.4e}  Diff: {diff_pct:6.2f}%")
                        elif isinstance(old_val, list) != isinstance(new_val, list):
                            print(f"  {stat_type}: FORMAT MISMATCH")
                        else:
                            diff_pct = abs(old_val - new_val) / (abs(old_val) + 1e-10) * 100
                            print(f"  {stat_type}: Old: {old_val:12.4e}  New: {new_val:12.4e}  Diff: {diff_pct:6.2f}%")


if __name__ == "__main__":
    main()
