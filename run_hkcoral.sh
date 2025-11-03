#!/bin/bash
set -euo pipefail

# Usage: ./run_hkcoral.sh [--phase train|evaluate|both] [--extra-args "..."]

phase="train"
extra_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)
      phase="$2"
      shift 2
      ;;
    --phase=*)
      phase="${1#*=}"
      shift
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

phase=$(echo "$phase" | tr '[:upper:]' '[:lower:]')

if [[ "$phase" == "train" || "$phase" == "both" ]]; then
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

  python train_hkcoral.py "${train_args[@]}" "${extra_args[@]}"
fi

if [[ "$phase" == "evaluate" || "$phase" == "both" ]]; then
  model_ids=(deeplabv3_resnet50 deeplabv3_resnet101 mask2former_swin_t mask2former_swin_s segformer_b0 segformer_b1)
  weights_dir=${WEIGHTS_DIR:-artifacts/hkcoral}
  eval_output_root=${EVAL_OUTPUT_DIR:-evaluation_outputs/hkcoral}
  eval_metrics_root=${EVAL_METRICS_DIR:-evaluation_outputs/hkcoral/metrics}
  heatmap_layer=${HEATMAP_LAYER:-encoder1.0}
  hkcoral_root=${HKCORAL_ROOT:-HKCoral}

  mkdir -p "${eval_output_root}" "${eval_metrics_root}"

  for model_id in "${model_ids[@]}"; do
    weight_path="${weights_dir}/${model_id}_best.pth"
    if [[ ! -f "${weight_path}" ]]; then
      echo "[WARN] Skipping ${model_id}: checkpoint not found at ${weight_path}" >&2
      continue
    fi

    model_output_dir="${eval_output_root}/${model_id}"
    model_metrics="${eval_metrics_root}/${model_id}_metrics.json"
    mkdir -p "${model_output_dir}"

    python evaluate.py \
      --dataset hkcoral \
      --dataset-arg root=${hkcoral_root} \
      --checkpoint "${weight_path}" \
      --model "${model_id}" \
      --num-classes 7 \
      --ignore-index 255 \
      --output-dir "${model_output_dir}" \
      --metrics-out "${model_metrics}" \
      --heatmap-layer "${heatmap_layer}" \
      "${extra_args[@]}"
  done
fi
