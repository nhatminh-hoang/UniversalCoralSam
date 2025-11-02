#!/bin/bash
set -euo pipefail

# Usage: ./run_hkcoral.sh [--extra-args "..."]
# Adjust the variables below to point to your dataset and output directories.

data_root=${DATA_ROOT:-HKCoral}
output_dir=${OUTPUT_DIR:-artifacts/hkcoral}
curve_dir=${CURVE_DIR:-artifacts/hkcoral/curves}
metrics_out=${METRICS_OUT:-artifacts/hkcoral/metrics.json}
batch_size=${BATCH_SIZE:-2}
num_workers=${NUM_WORKERS:-4}

mkdir -p "${output_dir}" "${curve_dir}"

train_args=(
  --data-root "${data_root}"
  --models deeplabv3 mask2former segformer
  --batch-size "${batch_size}"
  --num-workers "${num_workers}"
  --output-dir "${output_dir}"
  --curve-dir "${curve_dir}"
  --metrics-out "${metrics_out}"
)

python train_hkcoral.py "${train_args[@]}" "$@"
