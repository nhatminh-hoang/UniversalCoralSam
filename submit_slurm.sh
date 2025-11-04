#!/bin/bash
# Submit this script with: sbatch submit_slurm.sh

#SBATCH --job-name=hkcoral-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

module load python/3.10 2>/dev/null || true

export HKCORAL_ROOT=${HKCORAL_ROOT:-HKCoral}
export DATA_ROOT=${DATA_ROOT:-$HKCORAL_ROOT}
export OUTPUT_DIR=${OUTPUT_DIR:-artifacts/hkcoral}
export CURVE_DIR=${CURVE_DIR:-artifacts/hkcoral/curves}
export METRICS_OUT=${METRICS_OUT:-artifacts/hkcoral/metrics.json}
export BATCH_SIZE=${BATCH_SIZE:-2}
export EPOCHS=${EPOCHS:-100}
export NUM_WORKERS=${NUM_WORKERS:-4}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME=${HF_HOME:-${REPO_ROOT}/artifacts/hf_cache}
default_models="mask2former_swin_base mask2former_swin_large segformer_b2_cityscapes segformer_b5_cityscapes"
export MODELS="${MODELS:-$default_models}"

mkdir -p "$(dirname "${METRICS_OUT}")" "${OUTPUT_DIR}" "${CURVE_DIR}" "${HF_HOME}" logs

# Defer to the HKCoral helper script so the training/evaluation logic stays in one place.
bash run_hkcoral.sh --phase train "$@"
