#!/bin/bash
#SBATCH --job-name=fionet_pipeline
#SBATCH --qos=YOUR_QOS
#SBATCH --partition=YOUR_PARTITION
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --output=logs/%x.%j.out

set -euo pipefail
echo "Job started on $(date)"
echo "Running on node(s): $SLURM_NODELIST"
echo "Using temporary directory: $TMPDIR"

module purge
module load CUDA/12.1

echo "GPU info:"
nvidia-smi

#  configs/models/flower.yaml \ # flower
#  configs/models/unet_convnext.yaml \ # the-well convnext unet
#  configs/models/fno.yaml \ # the-well convnext unet
#  configs/models/scot.yaml
#  configs/models/avit.yaml

# Configuration variables
DATA_CONFIG="configs/data/turbulent_radiative_layer_2D.yaml"
MODEL_CONFIG="configs/models/flower.yaml"
TRAIN_CONFIG="configs/train.yaml"

# Check if this is validation mode
if [ "${VALIDATION_MODE:-false}" = "true" ]; then
    echo "Running VALIDATION"
    uv run python -m flowers.train \
    --data ${DATA_CONFIG} \
    --model ${MODEL_CONFIG} \
    --train ${TRAIN_CONFIG} \
    validation_mode=true \
    auto_resume=true
else
    echo "Running TRAINING"
    uv run python -m flowers.train \
    --data ${DATA_CONFIG} \
    --model ${MODEL_CONFIG} \
    --train ${TRAIN_CONFIG}
fi

echo "Job finished on $(date)"
