#!/usr/bin/env bash
set -euo pipefail

# NanoHorizon offline reference run.
#
# Edit this file first.
# Then run:
#   RUNPOD_API_KEY=... ./scripts/run_crafter_offline_reference.sh
#
# This script launches the full end-to-end offline baseline on RunPod:
#   1. start a local vLLM teacher for Qwen/Qwen3.5-27B on the pod
#   2. generate SFT rows from the starter seed prompts
#   3. filter rows with the heuristic reward
#   4. fine-tune Qwen/Qwen3.5-0.8B with TRL SFTTrainer
#   5. evaluate the resulting adapter

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/lib_runpod_webhook.sh"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Main knobs to change for your run.
RUN_NAME="${NANOHORIZON_RUN_NAME:-nanohorizon-offline-$(date -u +%Y%m%d-%H%M%S)}"
GIT_REPO="${NANOHORIZON_GIT_REPO:-https://github.com/synth-laboratories/nanohorizon.git}"
GIT_REF="${NANOHORIZON_GIT_REF:-main}"
CONFIG_PATH="${NANOHORIZON_OFFLINE_CONFIG:-configs/crafter_offline_qwen35_08b_1xa100_20min.yaml}"
GPU_TYPE="${NANOHORIZON_RUNPOD_GPU_TYPE:-NVIDIA A100 80GB PCIe}"
GPU_COUNT="${NANOHORIZON_RUNPOD_GPU_COUNT:-1}"
CONTAINER_DISK_GB="${NANOHORIZON_RUNPOD_CONTAINER_DISK_GB:-80}"
VOLUME_GB="${NANOHORIZON_RUNPOD_VOLUME_GB:-160}"
KEEPALIVE_AFTER="${NANOHORIZON_KEEPALIVE_AFTER:-0}"
WAIT_TIMEOUT_SECONDS="${NANOHORIZON_WAIT_TIMEOUT_SECONDS:-2400}"
COMPLETION_WEBHOOK_URL="$(resolve_runpod_completion_webhook)"
IMAGE_NAME="${NANOHORIZON_RUNPOD_IMAGE:-ghcr.io/synth-laboratories/nanohorizon-offline:latest}"
AUTO_INSTALL="${NANOHORIZON_AUTO_INSTALL:-0}"
START_LOCAL_TEACHER="${NANOHORIZON_START_LOCAL_TEACHER:-1}"
TEACHER_MODEL="${NANOHORIZON_TEACHER_MODEL:-Qwen/Qwen3.5-4B}"

# These env vars are forwarded into the pod when present.
declare -a FORWARDED_ENV=()
for key in GITHUB_TOKEN HF_TOKEN OPENAI_API_KEY NANOHORIZON_TEACHER_API_KEY NANOHORIZON_TEACHER_BASE_URL; do
  if [[ -n "${!key:-}" ]]; then
    FORWARDED_ENV+=(--env "$key=${!key}")
  fi
done

declare -a STOP_ARGS=(--auto-stop)
if [[ "$KEEPALIVE_AFTER" == "1" ]]; then
  STOP_ARGS=(--keepalive-after)
fi

TRAIN_CMD="cd /workspace/nanohorizon && NANOHORIZON_AUTO_INSTALL=${AUTO_INSTALL} NANOHORIZON_START_LOCAL_TEACHER=${START_LOCAL_TEACHER} NANOHORIZON_TEACHER_MODEL=${TEACHER_MODEL} bash scripts/run_crafter_offline_qwen35_08b_1xa100_20min.sh --config /workspace/nanohorizon/${CONFIG_PATH}"
if [[ "$AUTO_INSTALL" == "1" ]]; then
  SETUP_CMD="cd /workspace/nanohorizon && python3 -V && echo using public bootstrap runtime image"
else
  SETUP_CMD="cd /workspace/nanohorizon && python3 -V && echo using prebuilt offline runtime image"
fi

echo "NanoHorizon offline reference run"
echo "  started at: $STARTED_AT"
echo "  run name: $RUN_NAME"
echo "  git repo: $GIT_REPO"
echo "  git ref: $GIT_REF"
echo "  config: $CONFIG_PATH"
echo "  gpu: $GPU_TYPE x$GPU_COUNT"
echo "  image: $IMAGE_NAME"
echo "  auto install: $AUTO_INSTALL"
echo "  start local teacher: $START_LOCAL_TEACHER"
echo "  teacher model: $TEACHER_MODEL"
echo "  wait timeout seconds: $WAIT_TIMEOUT_SECONDS"
echo "  completion webhook: $COMPLETION_WEBHOOK_URL"

if (( ${#FORWARDED_ENV[@]} > 0 )); then
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m nanohorizon.runpod_training_launcher launch \
    --image-name "$IMAGE_NAME" \
    --name "$RUN_NAME" \
    --gpu-type-id "$GPU_TYPE" \
    --gpu-count "$GPU_COUNT" \
    --container-disk-gb "$CONTAINER_DISK_GB" \
    --volume-gb "$VOLUME_GB" \
    --support-public-ip \
    --wait-until-running \
    --wait-for-completion \
    --wait-timeout-seconds "$WAIT_TIMEOUT_SECONDS" \
    --completion-webhook-url "$COMPLETION_WEBHOOK_URL" \
    --git-repo "$GIT_REPO" \
    --git-ref "$GIT_REF" \
    --repo-dir nanohorizon \
    --setup-cmd "$SETUP_CMD" \
    --train-cmd "$TRAIN_CMD" \
    "${STOP_ARGS[@]}" \
    "${FORWARDED_ENV[@]}" \
    "$@"
else
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m nanohorizon.runpod_training_launcher launch \
    --image-name "$IMAGE_NAME" \
    --name "$RUN_NAME" \
    --gpu-type-id "$GPU_TYPE" \
    --gpu-count "$GPU_COUNT" \
    --container-disk-gb "$CONTAINER_DISK_GB" \
    --volume-gb "$VOLUME_GB" \
    --support-public-ip \
    --wait-until-running \
    --wait-for-completion \
    --wait-timeout-seconds "$WAIT_TIMEOUT_SECONDS" \
    --completion-webhook-url "$COMPLETION_WEBHOOK_URL" \
    --git-repo "$GIT_REPO" \
    --git-ref "$GIT_REF" \
    --repo-dir nanohorizon \
    --setup-cmd "$SETUP_CMD" \
    --train-cmd "$TRAIN_CMD" \
    "${STOP_ARGS[@]}" \
    "$@"
fi

ENDED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  ended at: $ENDED_AT"
