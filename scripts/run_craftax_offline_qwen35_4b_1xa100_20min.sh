#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${NANOHORIZON_OFFLINE_CONFIG:-configs/craftax_offline_reference.yaml}"
OUTPUT_DIR="${NANOHORIZON_OFFLINE_OUTPUT_DIR:-}"
TEACHER_MODEL="${NANOHORIZON_TEACHER_MODEL:-Qwen/Qwen3.5-9B}"
TEACHER_ENFORCE_EAGER="${NANOHORIZON_TEACHER_ENFORCE_EAGER:-1}"
TEACHER_STARTUP_ATTEMPTS="${NANOHORIZON_TEACHER_STARTUP_ATTEMPTS:-240}"
TEACHER_STARTUP_SLEEP_SECONDS="${NANOHORIZON_TEACHER_STARTUP_SLEEP_SECONDS:-2}"
DEPLOY_BEFORE_RUN="${NANOHORIZON_MODAL_DEPLOY_BEFORE_RUN:-1}"
APP_NAME="${NANOHORIZON_MODAL_OFFLINE_APP_NAME:-nanohorizon-craftax-offline}"
CODE_VERSION="${NANOHORIZON_CODE_VERSION:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"
export NANOHORIZON_MODAL_GPU_OFFLINE="${NANOHORIZON_MODAL_GPU_OFFLINE:-A100-40GB}"

cd "$ROOT"
export NANOHORIZON_TEACHER_MODEL="$TEACHER_MODEL"
source "$ROOT/scripts/lib_craftax_tunnel.sh"
trap nanohorizon_cleanup_craftax_tunnel EXIT
nanohorizon_open_craftax_tunnel_if_needed "$ROOT"
nanohorizon_start_modal_teacher_if_needed "$ROOT"
export NANOHORIZON_MODAL_OFFLINE_APP_NAME="$APP_NAME"
export NANOHORIZON_CODE_VERSION="$CODE_VERSION"
if [[ "$DEPLOY_BEFORE_RUN" == "1" ]]; then
  ./scripts/deploy_craftax_offline_modal_app.sh
fi
uv run --group modal python -m nanohorizon.shared.invoke_modal_offline \
  --config "$CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --teacher-model "$TEACHER_MODEL" \
  --teacher-api-key "${NANOHORIZON_TEACHER_API_KEY:-}" \
  --teacher-enforce-eager "$TEACHER_ENFORCE_EAGER" \
  --teacher-startup-attempts "$TEACHER_STARTUP_ATTEMPTS" \
  --teacher-startup-sleep-seconds "$TEACHER_STARTUP_SLEEP_SECONDS" \
  --craftax-container-url "${NANOHORIZON_CRAFTAX_CONTAINER_URL:-}" \
  --craftax-container-worker-token "${NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN:-}" \
  --teacher-inference-url "${NANOHORIZON_TEACHER_INFERENCE_URL:-}" \
  ${NANOHORIZON_MIN_TEACHER_REWARD:+--min-teacher-reward "$NANOHORIZON_MIN_TEACHER_REWARD"} \
  ${NANOHORIZON_MAX_TEACHER_ROWS:+--max-teacher-rows "$NANOHORIZON_MAX_TEACHER_ROWS"} \
  ${NANOHORIZON_FILTER_COLLECT_WOOD:+--filter-collect-wood "$NANOHORIZON_FILTER_COLLECT_WOOD"} \
  "$@"
status=$?
exit "$status"
