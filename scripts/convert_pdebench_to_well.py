#!/usr/bin/env python
"""
Convert PDEBench datasets to the-well compatible HDF5 format.

PDEBench datasets typically have structure:
- tensor: (n_samples, n_time, nx, ny, n_channels) or similar
- x-coordinate, y-coordinate, t-coordinate arrays

The Well format requires:
- Proper attributes (grid_type, n_spatial_dims, dataset_name, n_trajectories)
- dimensions group with spatial coords and time
- t0_fields, t1_fields, t2_fields groups for scalar/vector/tensor fields
- scalars group
- boundary_conditions group

Usage:
    uv run scripts/convert_pdebench_to_well.py
"""

import os
import argparse
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm


def inspect_pdebench_file(filepath: str) -> dict:
    """Inspect a PDEBench HDF5 file and return its structure."""
    info = {}
    with h5py.File(filepath, 'r') as f:
        info['keys'] = list(f.keys())
        info['attrs'] = dict(f.attrs)

        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                info[key] = {
                    'shape': f[key].shape,
                    'dtype': str(f[key].dtype),
                }
            else:
                info[key] = {'type': 'group', 'keys': list(f[key].keys())}

    return info


def convert_pdebench_2d_cfd(
    input_file: str,
    output_dir: str,
    dataset_name: str,
    boundary_type: str = "periodic",
    split: str = "train",
) -> None:
    """
    Convert PDEBench 2D CFD data to the-well format.

    PDEBench CFD format: (n_samples, n_time, nx, ny, 4)
    where channels are [density, pressure, Vx, Vy]
    """
    output_path = Path(output_dir) / dataset_name / "data" / split
    output_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_file, 'r') as src:
        # Get data array - PDEBench uses different key names
        if 'Vx' in src.keys():
            # Separate arrays format
            vx = src['Vx'][:]  # (n_samples, n_time, nx, ny) or (n_time, nx, ny)
            vy = src['Vy'][:]
            density = src['density'][:]
            pressure = src['pressure'][:]

            # Handle both shapes: Test files have (n_time, nx, ny), Train files have (n_samples, n_time, nx, ny)
            if len(vx.shape) == 3:
                # Test format: single trajectory (n_time, nx, ny)
                n_time, nx, ny = vx.shape
                n_samples = 1
                # Add samples dimension
                vx = vx[np.newaxis, ...]
                vy = vy[np.newaxis, ...]
                density = density[np.newaxis, ...]
                pressure = pressure[np.newaxis, ...]
            else:
                # Train format: (n_samples, n_time, nx, ny)
                n_samples = vx.shape[0]
                n_time = vx.shape[1]
                nx, ny = vx.shape[2], vx.shape[3]

            # Stack into single tensor
            data = np.stack([density, pressure, vx, vy], axis=-1)
        elif 'tensor' in src.keys():
            data = src['tensor'][:]
            n_samples, n_time, nx, ny, n_channels = data.shape
        else:
            raise ValueError(f"Unknown PDEBench format. Keys: {list(src.keys())}")

        # Get coordinates
        if 'x-coordinate' in src.keys():
            x_coord = src['x-coordinate'][:]
        else:
            x_coord = np.linspace(0, 1, nx)

        if 'y-coordinate' in src.keys():
            y_coord = src['y-coordinate'][:]
        else:
            y_coord = np.linspace(0, 1, ny)

        if 't-coordinate' in src.keys():
            t_coord = src['t-coordinate'][:]
        else:
            t_coord = np.linspace(0, 1, n_time)

        # Create output file
        original_name = Path(input_file).stem
        output_file = output_path / f"{original_name}.hdf5"

        with h5py.File(output_file, 'w') as dst:
            # Set root attributes
            dst.attrs['grid_type'] = 'cartesian'
            dst.attrs['n_spatial_dims'] = 2
            dst.attrs['dataset_name'] = dataset_name
            dst.attrs['n_trajectories'] = n_samples

            # Create dimensions group
            dims = dst.create_group('dimensions')
            dims.attrs['spatial_dims'] = ['x', 'y']

            # X coordinate
            x_ds = dims.create_dataset('x', data=x_coord)
            x_ds.attrs['sample_varying'] = False
            x_ds.attrs['time_varying'] = False

            # Y coordinate
            y_ds = dims.create_dataset('y', data=y_coord)
            y_ds.attrs['sample_varying'] = False
            y_ds.attrs['time_varying'] = False

            # Time coordinate (same for all samples)
            if len(t_coord.shape) == 1:
                time_ds = dims.create_dataset('time', data=t_coord)
                time_ds.attrs['sample_varying'] = False
            else:
                time_ds = dims.create_dataset('time', data=t_coord)
                time_ds.attrs['sample_varying'] = True
            time_ds.attrs['time_varying'] = False

            # Create scalar fields group (t0_fields)
            t0 = dst.create_group('t0_fields')
            t0.attrs['field_names'] = ['density', 'pressure']

            # Density field
            density_ds = t0.create_dataset('density', data=data[..., 0])
            density_ds.attrs['dim_varying'] = np.array([True, True])
            density_ds.attrs['sample_varying'] = True
            density_ds.attrs['time_varying'] = True
            density_ds.attrs['symmetric'] = False

            # Pressure field
            pressure_ds = t0.create_dataset('pressure', data=data[..., 1])
            pressure_ds.attrs['dim_varying'] = np.array([True, True])
            pressure_ds.attrs['sample_varying'] = True
            pressure_ds.attrs['time_varying'] = True
            pressure_ds.attrs['symmetric'] = False

            # Create vector fields group (t1_fields) for velocity
            t1 = dst.create_group('t1_fields')
            t1.attrs['field_names'] = ['velocity']

            # Velocity field (combine Vx, Vy)
            velocity = np.stack([data[..., 2], data[..., 3]], axis=-1)
            velocity_ds = t1.create_dataset('velocity', data=velocity)
            velocity_ds.attrs['dim_varying'] = np.array([True, True])
            velocity_ds.attrs['sample_varying'] = True
            velocity_ds.attrs['time_varying'] = True
            velocity_ds.attrs['symmetric'] = False

            # Create empty t2_fields group
            t2 = dst.create_group('t2_fields')
            t2.attrs['field_names'] = []

            # Create empty scalars group
            scalars = dst.create_group('scalars')
            scalars.attrs['field_names'] = []

            # Create boundary conditions group
            bc = dst.create_group('boundary_conditions')

            # X boundary
            bc_x = bc.create_group('x_boundary')
            bc_x.attrs['bc_type'] = boundary_type
            bc_x.attrs['associated_dims'] = ['x']
            bc_x.create_dataset('mask', data=np.ones((nx, ny), dtype=bool))

            # Y boundary
            bc_y = bc.create_group('y_boundary')
            bc_y.attrs['bc_type'] = boundary_type
            bc_y.attrs['associated_dims'] = ['y']
            bc_y.create_dataset('mask', data=np.ones((nx, ny), dtype=bool))

        print(f"  Created: {output_file}")
        print(f"    Samples: {n_samples}, Time steps: {n_time}, Resolution: {nx}x{ny}")


def convert_pdebench_ns_incomp(
    input_file: str,
    output_dir: str,
    dataset_name: str,
    boundary_type: str = "periodic",
    split: str = "train",
) -> None:
    """
    Convert PDEBench 2D incompressible NS data to the-well format.

    PDEBench NS incompressible format varies - check structure first.
    """
    output_path = Path(output_dir) / dataset_name / "data" / split
    output_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_file, 'r') as src:
        # Inspect structure
        keys = list(src.keys())

        # Different PDEBench NS formats
        if 'u' in keys:
            # Velocity only format (vorticity-streamfunction)
            u = src['u'][:]  # Could be (n_samples, n_time, nx, ny) or similar
            n_samples, n_time, nx, ny = u.shape

            # Create dummy pressure/density or extract if available
            data = u[..., np.newaxis]  # Just vorticity
            field_names = ['vorticity']
            is_scalar = True

        elif 'velocity' in keys:
            vel = src['velocity'][:]
            if len(vel.shape) == 5:
                n_samples, n_time, nx, ny, _ = vel.shape
            else:
                n_samples, n_time, nx, ny = vel.shape
                vel = vel[..., np.newaxis]
            data = vel
            field_names = ['velocity']
            is_scalar = False

        else:
            raise ValueError(f"Unknown NS format. Keys: {keys}")

        # Get coordinates
        x_coord = src.get('x-coordinate', np.linspace(0, 1, nx))[:]
        y_coord = src.get('y-coordinate', np.linspace(0, 1, ny))[:]
        t_coord = src.get('t-coordinate', np.linspace(0, 1, n_time))[:]

        original_name = Path(input_file).stem
        output_file = output_path / f"{original_name}.hdf5"

        with h5py.File(output_file, 'w') as dst:
            # Set root attributes
            dst.attrs['grid_type'] = 'cartesian'
            dst.attrs['n_spatial_dims'] = 2
            dst.attrs['dataset_name'] = dataset_name
            dst.attrs['n_trajectories'] = n_samples

            # Dimensions
            dims = dst.create_group('dimensions')
            dims.attrs['spatial_dims'] = ['x', 'y']

            x_ds = dims.create_dataset('x', data=x_coord if len(x_coord.shape) == 1 else x_coord.flatten())
            x_ds.attrs['sample_varying'] = False
            x_ds.attrs['time_varying'] = False

            y_ds = dims.create_dataset('y', data=y_coord if len(y_coord.shape) == 1 else y_coord.flatten())
            y_ds.attrs['sample_varying'] = False
            y_ds.attrs['time_varying'] = False

            time_ds = dims.create_dataset('time', data=t_coord if len(t_coord.shape) == 1 else t_coord[0])
            time_ds.attrs['sample_varying'] = False
            time_ds.attrs['time_varying'] = False

            # Fields
            if is_scalar:
                t0 = dst.create_group('t0_fields')
                t0.attrs['field_names'] = field_names

                for i, name in enumerate(field_names):
                    field_ds = t0.create_dataset(name, data=data[..., i] if data.shape[-1] > 1 else data[..., 0])
                    field_ds.attrs['dim_varying'] = np.array([True, True])
                    field_ds.attrs['sample_varying'] = True
                    field_ds.attrs['time_varying'] = True
                    field_ds.attrs['symmetric'] = False

                t1 = dst.create_group('t1_fields')
                t1.attrs['field_names'] = np.array([], dtype='S')
            else:
                t0 = dst.create_group('t0_fields')
                t0.attrs['field_names'] = np.array([], dtype='S')

                t1 = dst.create_group('t1_fields')
                t1.attrs['field_names'] = ['velocity']

                vel_ds = t1.create_dataset('velocity', data=data)
                vel_ds.attrs['dim_varying'] = np.array([True, True])
                vel_ds.attrs['sample_varying'] = True
                vel_ds.attrs['time_varying'] = True
                vel_ds.attrs['symmetric'] = False

            t2 = dst.create_group('t2_fields')
            t2.attrs['field_names'] = []

            scalars = dst.create_group('scalars')
            scalars.attrs['field_names'] = []

            # Boundary conditions
            bc = dst.create_group('boundary_conditions')
            bc_x = bc.create_group('x_boundary')
            bc_x.attrs['bc_type'] = boundary_type
            bc_x.attrs['associated_dims'] = ['x']
            bc_x.create_dataset('mask', data=np.ones((nx, ny), dtype=bool))

            bc_y = bc.create_group('y_boundary')
            bc_y.attrs['bc_type'] = boundary_type
            bc_y.attrs['associated_dims'] = ['y']
            bc_y.create_dataset('mask', data=np.ones((nx, ny), dtype=bool))

        print(f"  Created: {output_file}")
        print(f"    Samples: {n_samples}, Time steps: {n_time}, Resolution: {nx}x{ny}")


def convert_pdebench_grouped(
    input_file: str,
    output_dir: str,
    dataset_name: str,
    boundary_type: str = "periodic",
    split: str = "train",
    create_splits: bool = False,
    split_seed: int = 42,
) -> None:
    """
    Convert PDEBench grouped format (shallow-water, diffusion-reaction, etc.)
    where each sample is a separate group like '0000', '0001', etc.
    """
    with h5py.File(input_file, 'r') as src:
        # Get sample keys (numeric strings)
        sample_keys = sorted([k for k in src.keys() if k.isdigit()])
        n_samples = len(sample_keys)

        if n_samples == 0:
            raise ValueError("No numeric sample keys found")

        # Inspect first sample to determine structure
        first_sample = src[sample_keys[0]]
        data_array = first_sample['data'][:]

        # data shape is typically (n_time, nx, ny, n_channels) or (n_time, nx, ny)
        if len(data_array.shape) == 3:
            n_time, nx, ny = data_array.shape
            n_channels = 1
            data_array = data_array[..., np.newaxis]
        else:
            n_time, nx, ny, n_channels = data_array.shape

        # Get grid info
        if 'grid' in first_sample:
            grid = first_sample['grid']
            if 'x' in grid:
                x_coord = grid['x'][:]
                y_coord = grid['y'][:]
            else:
                # Sometimes grid is stored differently
                x_coord = np.linspace(0, 1, nx)
                y_coord = np.linspace(0, 1, ny)
            if 't' in grid:
                t_coord = grid['t'][:]
            else:
                t_coord = np.linspace(0, 1, n_time)
        else:
            x_coord = np.linspace(0, 1, nx)
            y_coord = np.linspace(0, 1, ny)
            t_coord = np.linspace(0, 1, n_time)

        # Determine field names based on dataset type
        if 'diff-react' in dataset_name.lower():
            field_names = ['activator', 'inhibitor'] if n_channels == 2 else [f'field_{i}' for i in range(n_channels)]
        elif 'shallow' in dataset_name.lower() or 'rdb' in dataset_name.lower():
            field_names = ['height'] if n_channels == 1 else [f'field_{i}' for i in range(n_channels)]
        elif 'darcy' in dataset_name.lower():
            field_names = ['pressure'] if n_channels == 1 else [f'field_{i}' for i in range(n_channels)]
        else:
            field_names = [f'field_{i}' for i in range(n_channels)]

        # Collect all samples
        all_data = np.zeros((n_samples, n_time, nx, ny, n_channels), dtype=np.float32)
        for i, key in enumerate(sample_keys):
            sample_data = src[key]['data'][:]
            if len(sample_data.shape) == 3:
                sample_data = sample_data[..., np.newaxis]
            all_data[i] = sample_data

        original_name = Path(input_file).stem

        # Create train/val/test splits if requested
        if create_splits and split == "train":  # Only split if original split was train
            np.random.seed(split_seed)
            indices = np.random.permutation(n_samples)

            n_train = int(0.8 * n_samples)
            n_val = int(0.1 * n_samples)

            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]

            splits_data = {
                'train': all_data[train_indices],
                'valid': all_data[val_indices],
                'test': all_data[test_indices],
            }

            for split_name, split_data in splits_data.items():
                output_path = Path(output_dir) / dataset_name / "data" / split_name
                output_path.mkdir(parents=True, exist_ok=True)
                output_file = output_path / f"{original_name}.hdf5"

                _write_grouped_hdf5(
                    output_file, split_data, dataset_name, field_names,
                    x_coord, y_coord, t_coord, boundary_type
                )
                print(f"  Created {split_name}: {output_file} ({len(split_data)} samples)")
            return

        # No splitting - just write to specified split
        output_path = Path(output_dir) / dataset_name / "data" / split
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{original_name}.hdf5"

        _write_grouped_hdf5(
            output_file, all_data, dataset_name, field_names,
            x_coord, y_coord, t_coord, boundary_type
        )
        print(f"  Created: {output_file}")
        print(f"    Samples: {n_samples}, Time steps: {n_time}, Resolution: {nx}x{ny}, Channels: {n_channels}")


def _write_grouped_hdf5(
    output_file: Path,
    data: np.ndarray,
    dataset_name: str,
    field_names: list,
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    t_coord: np.ndarray,
    boundary_type: str,
) -> None:
    """Helper to write grouped format data to HDF5."""
    n_samples = data.shape[0]
    n_time = data.shape[1]
    nx, ny = data.shape[2], data.shape[3]
    n_channels = data.shape[4]

    with h5py.File(output_file, 'w') as dst:
        dst.attrs['grid_type'] = 'cartesian'
        dst.attrs['n_spatial_dims'] = 2
        dst.attrs['dataset_name'] = dataset_name
        dst.attrs['n_trajectories'] = n_samples

        # Dimensions
        dims = dst.create_group('dimensions')
        dims.attrs['spatial_dims'] = ['x', 'y']

        x_ds = dims.create_dataset('x', data=x_coord.flatten() if len(x_coord.shape) > 1 else x_coord)
        x_ds.attrs['sample_varying'] = False
        x_ds.attrs['time_varying'] = False

        y_ds = dims.create_dataset('y', data=y_coord.flatten() if len(y_coord.shape) > 1 else y_coord)
        y_ds.attrs['sample_varying'] = False
        y_ds.attrs['time_varying'] = False

        time_ds = dims.create_dataset('time', data=t_coord.flatten() if len(t_coord.shape) > 1 else t_coord)
        time_ds.attrs['sample_varying'] = False
        time_ds.attrs['time_varying'] = False

        # Create fields
        t0 = dst.create_group('t0_fields')
        t0.attrs['field_names'] = field_names

        for i, name in enumerate(field_names):
            field_ds = t0.create_dataset(name, data=data[..., i])
            field_ds.attrs['dim_varying'] = np.array([True, True])
            field_ds.attrs['sample_varying'] = True
            field_ds.attrs['time_varying'] = True
            field_ds.attrs['symmetric'] = False

        t1 = dst.create_group('t1_fields')
        t1.attrs['field_names'] = []

        t2 = dst.create_group('t2_fields')
        t2.attrs['field_names'] = []

        scalars = dst.create_group('scalars')
        scalars.attrs['field_names'] = []

        # Boundary conditions
        bc = dst.create_group('boundary_conditions')
        bc_x = bc.create_group('x_boundary')
        bc_x.attrs['bc_type'] = boundary_type
        bc_x.attrs['associated_dims'] = ['x']
        bc_x.create_dataset('mask', data=np.ones((nx, ny), dtype=bool))

        bc_y = bc.create_group('y_boundary')
        bc_y.attrs['bc_type'] = boundary_type
        bc_y.attrs['associated_dims'] = ['y']
        bc_y.create_dataset('mask', data=np.ones((nx, ny), dtype=bool))


def convert_pdebench_ns_incom_full(
    input_file: str,
    output_dir: str,
    dataset_name: str,
    boundary_type: str = "periodic",
    split: str = "train",
) -> None:
    """
    Convert PDEBench NS incompressible with velocity/particles/force format.
    Shape: velocity (n_samples, n_time, nx, ny, 2)
    """
    output_path = Path(output_dir) / dataset_name / "data" / split
    output_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_file, 'r') as src:
        velocity = src['velocity'][:]  # (n_samples, n_time, nx, ny, 2)
        t_coord = src['t'][:]  # (n_samples, n_time) or (n_time,)

        n_samples, n_time, nx, ny, _ = velocity.shape

        # Get coordinates (assume unit domain)
        x_coord = np.linspace(0, 1, nx)
        y_coord = np.linspace(0, 1, ny)

        # Use first sample's time if sample-varying
        if len(t_coord.shape) > 1:
            t_coord = t_coord[0]

        original_name = Path(input_file).stem
        output_file = output_path / f"{original_name}.hdf5"

        with h5py.File(output_file, 'w') as dst:
            dst.attrs['grid_type'] = 'cartesian'
            dst.attrs['n_spatial_dims'] = 2
            dst.attrs['dataset_name'] = dataset_name
            dst.attrs['n_trajectories'] = n_samples

            # Dimensions
            dims = dst.create_group('dimensions')
            dims.attrs['spatial_dims'] = ['x', 'y']

            x_ds = dims.create_dataset('x', data=x_coord)
            x_ds.attrs['sample_varying'] = False
            x_ds.attrs['time_varying'] = False

            y_ds = dims.create_dataset('y', data=y_coord)
            y_ds.attrs['sample_varying'] = False
            y_ds.attrs['time_varying'] = False

            time_ds = dims.create_dataset('time', data=t_coord)
            time_ds.attrs['sample_varying'] = False
            time_ds.attrs['time_varying'] = False

            # Scalar fields (empty)
            t0 = dst.create_group('t0_fields')
            t0.attrs['field_names'] = np.array([], dtype='S')

            # Vector fields (velocity)
            t1 = dst.create_group('t1_fields')
            t1.attrs['field_names'] = ['velocity']

            vel_ds = t1.create_dataset('velocity', data=velocity)
            vel_ds.attrs['dim_varying'] = np.array([True, True])
            vel_ds.attrs['sample_varying'] = True
            vel_ds.attrs['time_varying'] = True
            vel_ds.attrs['symmetric'] = False

            t2 = dst.create_group('t2_fields')
            t2.attrs['field_names'] = []

            scalars = dst.create_group('scalars')
            scalars.attrs['field_names'] = []

            # Boundary conditions
            bc = dst.create_group('boundary_conditions')
            bc_x = bc.create_group('x_boundary')
            bc_x.attrs['bc_type'] = boundary_type
            bc_x.attrs['associated_dims'] = ['x']
            bc_x.create_dataset('mask', data=np.ones((nx, ny), dtype=bool))

            bc_y = bc.create_group('y_boundary')
            bc_y.attrs['bc_type'] = boundary_type
            bc_y.attrs['associated_dims'] = ['y']
            bc_y.create_dataset('mask', data=np.ones((nx, ny), dtype=bool))

        print(f"  Created: {output_file}")
        print(f"    Samples: {n_samples}, Time steps: {n_time}, Resolution: {nx}x{ny}")


def convert_pdebench_generic(
    input_file: str,
    output_dir: str,
    dataset_name: str,
    boundary_type: str = "periodic",
    split: str = "train",
    create_splits: bool = False,
    split_seed: int = 42,
) -> None:
    """
    Generic converter that inspects PDEBench file and converts accordingly.
    """
    with h5py.File(input_file, 'r') as src:
        keys = list(src.keys())
        print(f"  Inspecting {input_file}")
        print(f"    Keys: {keys[:10]}{'...' if len(keys) > 10 else ''}")

        # Check for grouped format first (numeric keys like '0000', '0001')
        numeric_keys = [k for k in keys if k.isdigit()]
        if len(numeric_keys) > 10:  # Likely grouped format
            print("    Detected: Grouped samples format")
            convert_pdebench_grouped(input_file, output_dir, dataset_name, boundary_type, split, create_splits, split_seed)
            return

        # Detect format and call appropriate converter
        if 'Vx' in keys or ('density' in keys and 'pressure' in keys):
            print("    Detected: CFD format")
            convert_pdebench_2d_cfd(input_file, output_dir, dataset_name, boundary_type, split)
        elif 'velocity' in keys and 'particles' in keys:
            print("    Detected: NS incompressible (velocity/particles) format")
            convert_pdebench_ns_incom_full(input_file, output_dir, dataset_name, boundary_type, split)
        elif 'u' in keys or 'velocity' in keys:
            print("    Detected: NS incompressible format")
            convert_pdebench_ns_incomp(input_file, output_dir, dataset_name, boundary_type, split)
        elif 'tensor' in keys:
            print("    Detected: Generic tensor format")
            tensor_shape = src['tensor'].shape
            if len(tensor_shape) == 5:
                n_channels = tensor_shape[-1]
                if n_channels == 4:
                    print("    Assuming CFD (density, pressure, Vx, Vy)")
                    convert_pdebench_2d_cfd(input_file, output_dir, dataset_name, boundary_type, split)
                else:
                    print(f"    {n_channels} channels, using generic converter")
                    convert_generic_tensor(input_file, output_dir, dataset_name, boundary_type, split, create_splits=create_splits, split_seed=split_seed)
            elif len(tensor_shape) == 4:
                # Shape: (n_samples, n_time, nx, ny)
                print(f"    4D tensor shape {tensor_shape}, using generic converter")
                convert_generic_tensor(input_file, output_dir, dataset_name, boundary_type, split, create_splits=create_splits, split_seed=split_seed)
            else:
                print(f"    WARNING: Unknown tensor shape {tensor_shape}, skipping")
        else:
            print(f"    WARNING: Unknown format with keys {keys[:5]}")
            # Try generic approach
            for key in keys:
                if isinstance(src[key], h5py.Dataset) and len(src[key].shape) >= 4:
                    print(f"    Attempting generic conversion with '{key}'")
                    convert_generic_tensor(input_file, output_dir, dataset_name, boundary_type, split, data_key=key, create_splits=create_splits, split_seed=split_seed)
                    break


def convert_generic_tensor(
    input_file: str,
    output_dir: str,
    dataset_name: str,
    boundary_type: str = "periodic",
    split: str = "train",
    data_key: str = "tensor",
    create_splits: bool = False,
    split_seed: int = 42,
) -> None:
    """Fallback converter for generic tensor data."""
    with h5py.File(input_file, 'r') as src:
        data = src[data_key][:]

        # Assume (n_samples, n_time, nx, ny[, n_channels])
        if len(data.shape) == 4:
            n_samples, n_time, nx, ny = data.shape
            data = data[..., np.newaxis]
        else:
            n_samples, n_time, nx, ny, n_channels = data.shape

        # Handle single timestep datasets (like DarcyFlow)
        # Check if there's an input field (nu) to use as constant field
        constant_field_data = None
        constant_field_name = None
        if n_time == 1:
            if 'nu' in src.keys():
                # DarcyFlow-style: nu (constant permeability input) -> tensor (pressure output)
                # Treat nu as a constant field, not a time-varying input
                constant_field_data = src['nu'][:]  # (n_samples, nx, ny)
                constant_field_name = 'permeability'
                # data remains as (n_samples, 1, nx, ny, n_channels)
            else:
                raise ValueError(
                    f"Dataset has only 1 timestep but no 'nu' field found for input-output pairing. "
                    f"Available keys: {list(src.keys())}. Cannot convert single-timestep data without input field."
                )

        x_coord = np.linspace(0, 1, nx)
        y_coord = np.linspace(0, 1, ny)
        t_coord = np.linspace(0, 1, n_time)

        original_name = Path(input_file).stem

        # Create train/val/test splits if requested
        if create_splits and split == "train":  # Only split if original split was train
            np.random.seed(split_seed)
            indices = np.random.permutation(n_samples)

            n_train = int(0.8 * n_samples)
            n_val = int(0.1 * n_samples)

            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]

            splits_data = {
                'train': data[train_indices],
                'valid': data[val_indices],
                'test': data[test_indices],
            }

            for split_name, split_data in splits_data.items():
                output_path = Path(output_dir) / dataset_name / "data" / split_name
                output_path.mkdir(parents=True, exist_ok=True)
                output_file = output_path / f"{original_name}.hdf5"

                # Split constant field data if present
                if constant_field_data is not None:
                    if split_name == 'train':
                        split_constant = constant_field_data[train_indices]
                    elif split_name == 'valid':
                        split_constant = constant_field_data[val_indices]
                    else:
                        split_constant = constant_field_data[test_indices]
                else:
                    split_constant = None

                _write_generic_hdf5(
                    output_file, split_data, dataset_name,
                    x_coord, y_coord, t_coord, boundary_type,
                    constant_field_data=split_constant,
                    constant_field_name=constant_field_name,
                )
                print(f"  Created {split_name}: {output_file} ({len(split_data)} samples)")
            return

        # No splitting - just write to specified split
        output_path = Path(output_dir) / dataset_name / "data" / split
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{original_name}.hdf5"

        _write_generic_hdf5(
            output_file, data, dataset_name,
            x_coord, y_coord, t_coord, boundary_type,
            constant_field_data=constant_field_data,
            constant_field_name=constant_field_name,
        )
        print(f"  Created: {output_file} ({n_samples} samples)")


def _write_generic_hdf5(
    output_file: Path,
    data: np.ndarray,
    dataset_name: str,
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    t_coord: np.ndarray,
    boundary_type: str,
    constant_field_data: np.ndarray = None,
    constant_field_name: str = None,
) -> None:
    """Helper to write generic tensor data to HDF5."""
    n_samples = data.shape[0]
    nx, ny = data.shape[2], data.shape[3]

    with h5py.File(output_file, 'w') as dst:
        dst.attrs['grid_type'] = 'cartesian'
        dst.attrs['n_spatial_dims'] = 2
        dst.attrs['dataset_name'] = 'pdebench-' + dataset_name
        dst.attrs['n_trajectories'] = n_samples

        dims = dst.create_group('dimensions')
        dims.attrs['spatial_dims'] = ['x', 'y']

        x_ds = dims.create_dataset('x', data=x_coord)
        x_ds.attrs['sample_varying'] = False
        x_ds.attrs['time_varying'] = False

        y_ds = dims.create_dataset('y', data=y_coord)
        y_ds.attrs['sample_varying'] = False
        y_ds.attrs['time_varying'] = False

        time_ds = dims.create_dataset('time', data=t_coord)
        time_ds.attrs['sample_varying'] = False
        time_ds.attrs['time_varying'] = False

        # Put all channels as scalar fields
        t0 = dst.create_group('t0_fields')
        field_names = [f'field_{i}' for i in range(data.shape[-1])]

        # Add constant field to field_names if present
        if constant_field_data is not None and constant_field_name is not None:
            field_names = [constant_field_name] + field_names

        t0.attrs['field_names'] = field_names

        # Write constant field first if present (no time dimension)
        if constant_field_data is not None and constant_field_name is not None:
            constant_ds = t0.create_dataset(constant_field_name, data=constant_field_data)
            constant_ds.attrs['dim_varying'] = np.array([True, True])
            constant_ds.attrs['sample_varying'] = True
            constant_ds.attrs['time_varying'] = False  # This is the key difference!
            constant_ds.attrs['symmetric'] = False

        # Write time-varying fields
        for i, name in enumerate([f'field_{i}' for i in range(data.shape[-1])]):
            field_ds = t0.create_dataset(name, data=data[..., i])
            field_ds.attrs['dim_varying'] = np.array([True, True])
            field_ds.attrs['sample_varying'] = True
            field_ds.attrs['time_varying'] = True
            field_ds.attrs['symmetric'] = False

        t1 = dst.create_group('t1_fields')
        t1.attrs['field_names'] = []

        t2 = dst.create_group('t2_fields')
        t2.attrs['field_names'] = []

        scalars = dst.create_group('scalars')
        scalars.attrs['field_names'] = []

        bc = dst.create_group('boundary_conditions')
        bc_x = bc.create_group('x_boundary')
        bc_x.attrs['bc_type'] = boundary_type
        bc_x.attrs['associated_dims'] = ['x']
        bc_x.create_dataset('mask', data=np.ones((nx, ny), dtype=bool))

        bc_y = bc.create_group('y_boundary')
        bc_y.attrs['bc_type'] = boundary_type
        bc_y.attrs['associated_dims'] = ['y']
        bc_y.create_dataset('mask', data=np.ones((nx, ny), dtype=bool))


def main():
    parser = argparse.ArgumentParser(description="Convert PDEBench datasets to the-well format")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/PDEBench",
        help="Input PDEBench data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/PDEBench-well",
        help="Output directory for converted data",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset to convert (e.g., '2D/CFD/2D_Train_Rand'). If not specified, converts all.",
    )
    parser.add_argument(
        "--boundary-type",
        type=str,
        default="periodic",
        choices=["periodic", "open", "wall"],
        help="Default boundary condition type",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only inspect files without converting",
    )
    parser.add_argument(
        "--create-splits",
        action="store_true",
        help="Create train/valid/test splits (80/10/10) from training data",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for train/valid/test splitting",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("PDEBench to The-Well Converter")
    print("=" * 80)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()

    # Find all HDF5 files (excluding 1D data)
    h5_files = list(input_dir.rglob("*.h5")) + list(input_dir.rglob("*.hdf5"))
    h5_files = [f for f in h5_files if '/1D/' not in str(f)]

    if args.dataset:
        # Filter to specific dataset
        h5_files = [f for f in h5_files if args.dataset in str(f)]

    print(f"Found {len(h5_files)} HDF5 files (excluding 1D)")
    print()

    if args.inspect_only:
        for f in h5_files:
            print(f"\n{f}")
            info = inspect_pdebench_file(str(f))
            for key, val in info.items():
                print(f"  {key}: {val}")
        return

    # Convert each file
    for h5_file in tqdm(h5_files, desc="Converting"):
        # Derive dataset name from path - group by equation type, not individual directories
        rel_path = h5_file.relative_to(input_dir)
        parts = rel_path.parts

        # Create meaningful dataset name based on equation type
        # Examples: 2D/CFD/*, 2D/NS_incom/*, 3D/Train/*, etc.
        if len(parts) >= 2:
            # Use dimension + equation type (e.g., "2D_CFD", "2D_NS_incom", "3D_CFD")
            dataset_name = f"{parts[0]}_{parts[1]}"
        else:
            dataset_name = h5_file.stem

        dataset_name = dataset_name.replace("-", "_")

        # Determine split from path (not just filename)
        rel_path_lower = str(rel_path).lower()
        fname = h5_file.stem.lower()
        if "train" in rel_path_lower or "train" in fname:
            split = "train"
        elif "test" in rel_path_lower or "test" in fname:
            split = "test"
        elif "valid" in rel_path_lower or "valid" in fname:
            split = "valid"
        else:
            split = "train"  # Default

        # Determine boundary type based on dataset type (from PDEBench docs)
        # Map: periodic, open (Dirichlet/Cauchy), wall (Neumann/no-flow)
        rel_path_str = str(h5_file).lower()

        if "diff-react" in rel_path_str or "diffusion-reaction" in rel_path_str:
            boundary_type = "wall"  # Neumann (no-flow)
        elif "darcy" in rel_path_str:
            boundary_type = "open"  # Dirichlet
        elif "ns_incom" in rel_path_str or "incomp" in rel_path_str:
            boundary_type = "open"  # Dirichlet
        elif "shallow" in rel_path_str or "rdb" in rel_path_str:
            boundary_type = "wall"  # Neumann
        elif "cfd" in rel_path_str or "navier-stokes" in rel_path_str:
            boundary_type = "periodic"  # Compressible NS 2D/3D
        elif "periodic" in fname:
            boundary_type = "periodic"
        else:
            boundary_type = args.boundary_type

        print(f"\nConverting: {h5_file}")
        print(f"  -> Dataset: {dataset_name}, Split: {split}, BC: {boundary_type}")

        try:
            # Pass create_splits flag to converters
            if 'generic' in str(convert_pdebench_generic):
                convert_pdebench_generic(
                    str(h5_file),
                    str(output_dir),
                    dataset_name,
                    boundary_type,
                    split,
                    args.create_splits,
                    args.split_seed,
                )
            else:
                convert_pdebench_generic(
                    str(h5_file),
                    str(output_dir),
                    dataset_name,
                    boundary_type,
                    split,
                )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Conversion complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
