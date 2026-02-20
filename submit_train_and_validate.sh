#!/bin/bash
# submit_both.sh

# Submit training job and capture its ID
TRAIN_JOB=$(sbatch --parsable --time=24:00:00 train.sbatch)
echo "Submitted training job: ${TRAIN_JOB}"

# Submit validation job that waits for training to finish
VAL_JOB=$(sbatch --parsable --time=2:00:00 --dependency=afterany:${TRAIN_JOB} \
    --export=ALL,VALIDATION_MODE=true train.sbatch)
echo "Submitted validation job: ${VAL_JOB} (depends on ${TRAIN_JOB})"
