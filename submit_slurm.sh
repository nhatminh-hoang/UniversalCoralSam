#!/bin/bash
# Submit this script with: sbatch submit_slurm.sh

#SBATCH --job-name=coral-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

module load python/3.10 2>/dev/null || true

export IMAGES_DIR=${IMAGES_DIR:-data/images}
export MASKS_DIR=${MASKS_DIR:-data/masks}
export NUM_CLASSES=${NUM_CLASSES:-5}
export EPOCHS=${EPOCHS:-10}
export BATCH_SIZE=${BATCH_SIZE:-4}

python train.py \
  --images "$IMAGES_DIR" \
  --masks "$MASKS_DIR" \
  --num-classes "$NUM_CLASSES" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --curve-path "training_curve.png"

