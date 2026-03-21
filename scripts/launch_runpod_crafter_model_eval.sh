#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/lib_runpod_webhook.sh"
source "$ROOT/scripts/lib_runpod_gpu.sh"
GIT_REPO="${NANOHORIZON_GIT_REPO:-https://github.com/synth-laboratories/nanohorizon.git}"
GIT_REF="${NANOHORIZON_GIT_REF:-main}"
BASE_MODEL="${NANOHORIZON_EVAL_BASE_MODEL:-Qwen/Qwen3.5-4B}"
WAIT_TIMEOUT_SECONDS="${NANOHORIZON_WAIT_TIMEOUT_SECONDS:-1800}"
COMPLETION_WEBHOOK_URL="$(resolve_runpod_completion_webhook)"
IMAGE_NAME="${NANOHORIZON_RUNPOD_IMAGE:-ghcr.io/synth-laboratories/nanohorizon-eval:latest}"

FORWARDED_ENV=()
for key in GITHUB_TOKEN HF_TOKEN; do
  if [[ -n "${!key:-}" ]]; then
    FORWARDED_ENV+=(--env "$key=${!key}")
  fi
done

TRAIN_ENV=(
  "NANOHORIZON_EVAL_BASE_MODEL=$BASE_MODEL"
)
if [[ -n "${NANOHORIZON_EVAL_ADAPTER_DIR:-}" ]]; then
  TRAIN_ENV+=("NANOHORIZON_EVAL_ADAPTER_DIR=${NANOHORIZON_EVAL_ADAPTER_DIR}")
fi

TRAIN_PREFIX=""
for item in "${TRAIN_ENV[@]}"; do
  TRAIN_PREFIX+="$item "
done

nanoh_runpod_gpu_load
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m nanohorizon.runpod_training_launcher launch \
  --image-name "$IMAGE_NAME" \
  --name "nanohorizon-eval-$(date -u +%Y%m%d-%H%M%S)" \
  "${NANOH_RUNPOD_GPU_ARGS[@]}" \
  --gpu-count "${NANOHORIZON_RUNPOD_GPU_COUNT:-1}" \
  --container-disk-gb "${NANOHORIZON_RUNPOD_CONTAINER_DISK_GB:-80}" \
  --volume-gb "${NANOHORIZON_RUNPOD_VOLUME_GB:-120}" \
  --support-public-ip \
  --git-repo "$GIT_REPO" \
  --git-ref "$GIT_REF" \
  --repo-dir nanohorizon \
  --setup-cmd "cd /workspace/nanohorizon && python3 -V && echo using prebuilt eval runtime image" \
  --train-cmd "cd /workspace/nanohorizon && ${TRAIN_PREFIX}bash scripts/run_crafter_model_eval.sh" \
  --wait-until-running \
  --wait-for-completion \
  --wait-timeout-seconds "$WAIT_TIMEOUT_SECONDS" \
  --completion-webhook-url "$COMPLETION_WEBHOOK_URL" \
  --auto-stop \
  ${FORWARDED_ENV[@]+"${FORWARDED_ENV[@]}"} \
  "$@"
