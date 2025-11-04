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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${HF_HOME:-}" ]]; then
  export HF_HOME="${script_dir}/artifacts/hf_cache"
fi
mkdir -p "${HF_HOME}"

default_models=(mask2former_swin_base mask2former_swin_large segformer_b2_cityscapes segformer_b5_cityscapes)
if [[ -n "${MODELS:-}" ]]; then
  read -r -a train_models <<< "${MODELS}"
else
  train_models=("${default_models[@]}")
fi

if [[ -n "${EVAL_MODELS:-}" ]]; then
  read -r -a eval_models <<< "${EVAL_MODELS}"
else
  eval_models=("${train_models[@]}")
fi

if [[ "$phase" == "train" || "$phase" == "both" ]]; then
  data_root=${DATA_ROOT:-HKCoral}
  output_dir=${OUTPUT_DIR:-artifacts/hkcoral}
  curve_dir=${CURVE_DIR:-artifacts/hkcoral/curves}
  metrics_out=${METRICS_OUT:-artifacts/hkcoral/metrics.json}
  batch_size=${BATCH_SIZE:-2}
  num_workers=${NUM_WORKERS:-4}
  epochs=${EPOCHS:-100}
  lr=${LR:-1e-4}
  weight_decay=${WEIGHT_DECAY:-1e-2}
  warmup_epochs=${WARMUP_EPOCHS:-10}
  dropout=${DROPOUT:-0.0}

  mkdir -p "${output_dir}" "${curve_dir}"

  train_args=(
    --data-root "${data_root}"
    --batch-size "${batch_size}"
    --num-workers "${num_workers}"
    --epochs "${epochs}"
    --lr "${lr}"
    --weight-decay "${weight_decay}"
    --warmup-epochs "${warmup_epochs}"
    --dropout "${dropout}"
    --output-dir "${output_dir}"
    --curve-dir "${curve_dir}"
    --metrics-out "${metrics_out}"
  )
  train_args+=(--models)
  train_args+=("${train_models[@]}")

  python train_hkcoral.py "${train_args[@]}" "${extra_args[@]}"
fi

if [[ "$phase" == "evaluate" || "$phase" == "both" ]]; then
  model_ids=("${eval_models[@]}")
  weights_dir=${WEIGHTS_DIR:-artifacts/hkcoral}
  eval_output_root=${EVAL_OUTPUT_DIR:-evaluation_outputs/hkcoral}
  eval_metrics_root=${EVAL_METRICS_DIR:-evaluation_outputs/hkcoral/metrics}
  heatmap_layer=${HEATMAP_LAYER:-}
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

    eval_args=(
      --dataset hkcoral
      --dataset-arg root=${hkcoral_root}
      --dataset-arg split=test
      --checkpoint "${weight_path}"
      --model "${model_id}"
      --num-classes 7
      --ignore-index 255
      --output-dir "${model_output_dir}"
      --metrics-out "${model_metrics}"
    )

    if [[ -n "${heatmap_layer}" ]]; then
      eval_args+=(--heatmap-layer "${heatmap_layer}")
    fi

    python evaluate.py "${eval_args[@]}"
  done
fi
