#!/bin/bash

# PDEBench Full Dataset Download Script
# This script downloads ALL PDEBench datasets into data/PDEBench/
# WARNING: Total size is approximately 3.2 TB - make sure you have enough space!

set -e  # Exit on error

# Install PDEBench dependencies (without data generation extras)
echo "Installing PDEBench dependencies..."
uv sync --no-all-extras
echo ""

# Set the root folder for downloads
ROOT_FOLDER="data/PDEBench"
SCRIPT_DIR="PDEBench/pdebench/data_download"

# Store the flowers root directory
WARPSPEED_ROOT="$(pwd)"

# Create the root folder if it doesn't exist
mkdir -p "$ROOT_FOLDER"

echo "=========================================="
echo "PDEBench Dataset Download Script"
echo "=========================================="
echo "Root folder: $ROOT_FOLDER"
echo "WARNING: Total download size ~3.2 TB"
echo "=========================================="
echo ""

# List of all available PDEs with their sizes
declare -a PDES=(
    #"advection:47GB"
    #"burgers:93GB"
    #"1d_cfd:88GB"
    #"diff_sorp:4GB"
    #"1d_reacdiff:62GB"
    #"2d_reacdiff:13GB"
    #"2d_cfd:551GB"
    #"3d_cfd:285GB"
    #"darcy:6.2GB"
    "ns_incom:2.3TB"
    "swe:6.2GB"
)

# Download each dataset
for pde_info in "${PDES[@]}"; do
    IFS=':' read -r pde_name size <<< "$pde_info"
    echo "=========================================="
    echo "Downloading: $pde_name (Size: $size)"
    echo "=========================================="
    (cd "$SCRIPT_DIR" && "$WARPSPEED_ROOT/.venv/bin/python" download_direct.py --root_folder "../../../$ROOT_FOLDER" --pde_name "$pde_name")
    echo ""
    echo "✓ Completed: $pde_name"
    echo ""
done

echo "=========================================="
echo "All PDEBench datasets downloaded!"
echo "Location: $ROOT_FOLDER"
echo "=========================================="
