#!/usr/bin/env python
"""
Download datasets from The Well with automatic corruption checking and retry.

This script:
1. Downloads the specified dataset
2. Verifies all HDF5 files are valid (not corrupted/truncated)
3. Re-downloads corrupted files automatically
4. Retries up to MAX_RETRIES times per split
"""

import os
from pathlib import Path
import h5py
from the_well.utils.download import well_download

# Configuration
BASE_PATH = os.path.join(os.path.dirname(__file__), "../data", "the-well")
DATASET_NAME = "supernova_explosion_64"  # Change this to download different datasets
MAX_RETRIES = 5  # Maximum number of retry attempts per split


def check_hdf5_files(dataset_path, split):
    """
    Check all HDF5 files in a split for corruption.

    Returns:
        list: List of corrupted file paths
    """
    split_path = Path(dataset_path) / "data" / split

    if not split_path.exists():
        print(f"    Warning: {split} directory doesn't exist")
        return []

    files = sorted(split_path.glob("*.hdf5"))

    if not files:
        print(f"    Warning: No HDF5 files found in {split}")
        return []

    corrupted = []

    print(f"    Checking {len(files)} files in {split}...")

    for f in files:
        try:
            with h5py.File(f, 'r') as hf:
                # Try to read keys to verify file is valid
                _ = list(hf.keys())
        except (OSError, Exception) as e:
            print(f"      ✗ {f.name}: CORRUPTED")

            # Check if it's an HTML error page or truncated
            with open(f, 'rb') as corrupted_file:
                header = corrupted_file.read(100)
                if b'<html>' in header or b'503' in header:
                    print(f"        → HTML error page (503)")
                else:
                    print(f"        → Truncated or invalid: {str(e)[:80]}")

            corrupted.append(f)

    if corrupted:
        print(f"      Found {len(corrupted)}/{len(files)} corrupted files")
    else:
        print(f"      ✓ All {len(files)} files valid")

    return corrupted


def download_with_verification(base_path, dataset_name, max_retries=5):
    """
    Download dataset and verify files, retrying corrupted splits.

    Args:
        base_path: Base path for downloads
        dataset_name: Name of the dataset
        max_retries: Maximum retry attempts per split

    Returns:
        bool: True if all files are valid, False otherwise
    """
    dataset_path = Path(base_path) / "datasets" / dataset_name

    print("=" * 80)
    print(f"DOWNLOADING: {dataset_name}")
    print("=" * 80)

    # Initial download
    print("\nStep 1: Initial download")
    print("-" * 80)
    try:
        well_download(
            base_path=base_path,
            dataset=dataset_name,
            split=None,  # Download all splits
            first_only=False,
            parallel=True,
        )
        print("✓ Initial download complete")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

    # Check and retry corrupted files
    print("\nStep 2: Verification and retry")
    print("-" * 80)

    splits = ["train", "valid", "test"]
    retry_counts = {split: 0 for split in splits}

    all_valid = False
    overall_attempt = 0

    while not all_valid and overall_attempt < max_retries:
        overall_attempt += 1

        print(f"\n  Verification attempt {overall_attempt}/{max_retries}")

        splits_to_retry = []

        for split in splits:
            corrupted = check_hdf5_files(dataset_path, split)

            if corrupted:
                splits_to_retry.append(split)
                retry_counts[split] += 1

        if not splits_to_retry:
            all_valid = True
            print("\n  ✓ All files are valid!")
            break

        # Retry corrupted splits
        print(f"\n  Found corrupted files in: {', '.join(splits_to_retry)}")

        for split in splits_to_retry:
            if retry_counts[split] >= max_retries:
                print(f"    ✗ {split}: Max retries reached, giving up")
                continue

            print(f"    Re-downloading {split} split (attempt {retry_counts[split]}/{max_retries})...")

            # Delete corrupted files
            corrupted = check_hdf5_files(dataset_path, split)
            for f in corrupted:
                print(f"      Deleting {f.name}")
                f.unlink()

            # Re-download
            try:
                well_download(
                    base_path=base_path,
                    dataset=dataset_name,
                    split=split,
                    first_only=False,
                    parallel=True,
                )
                print(f"      ✓ Re-download complete")
            except Exception as e:
                print(f"      ✗ Re-download failed: {e}")

    # Final verification
    print("\n" + "=" * 80)
    print("FINAL VERIFICATION")
    print("=" * 80)

    final_corrupted = []
    for split in splits:
        corrupted = check_hdf5_files(dataset_path, split)
        final_corrupted.extend(corrupted)

    if final_corrupted:
        print(f"\n✗ DOWNLOAD INCOMPLETE: {len(final_corrupted)} files still corrupted")
        print("  Corrupted files:")
        for f in final_corrupted:
            print(f"    - {f.relative_to(dataset_path)}")
        return False
    else:
        print("\n✓ SUCCESS: All files downloaded and verified!")
        return True


if __name__ == "__main__":
    print("=" * 80)
    print("THE WELL DATASET DOWNLOADER")
    print("=" * 80)
    print(f"\nDataset: {DATASET_NAME}")
    print(f"Destination: {BASE_PATH}/datasets/{DATASET_NAME}")
    print(f"Max retries per split: {MAX_RETRIES}")
    print("\nThis may take a while depending on dataset size...")
    print("Press Ctrl+C to stop - downloads will resume where they left off.")
    print("")

    success = download_with_verification(BASE_PATH, DATASET_NAME, MAX_RETRIES)

    if success:
        print("\n" + "=" * 80)
        print("DOWNLOAD COMPLETE!")
        print("=" * 80)
        print(f"Dataset location: {BASE_PATH}/datasets/{DATASET_NAME}")
    else:
        print("\n" + "=" * 80)
        print("DOWNLOAD FAILED")
        print("=" * 80)
        print("Some files could not be downloaded successfully.")
        print("You may want to:")
        print("  1. Check your internet connection")
        print("  2. Try again later (server may be having issues)")
        print("  3. Check available disk space")
        exit(1)
