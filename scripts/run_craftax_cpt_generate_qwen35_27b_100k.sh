#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${NANOHORIZON_CPT_DATA_CONFIG:-configs/craftax_cpt_data_qwen35_27b_100k_fast.yaml}"
OUTPUT_DIR="${NANOHORIZON_CPT_DATA_OUTPUT_DIR:-$ROOT/artifacts/cpt_data_qwen35_27b_100k_$(date -u +%Y%m%dT%H%M%SZ)}"
USE_LOCAL_DIRECT="${NANOHORIZON_CPT_DATA_USE_LOCAL_DIRECT:-0}"

export NANOHORIZON_MODAL_GPU_TEACHER="${NANOHORIZON_MODAL_GPU_TEACHER:-B200}"
export NANOHORIZON_TEACHER_MODEL="${NANOHORIZON_TEACHER_MODEL:-Qwen/Qwen3.5-27B}"
export NANOHORIZON_TEACHER_MAX_MODEL_LEN="${NANOHORIZON_TEACHER_MAX_MODEL_LEN:-16384}"
export NANOHORIZON_TEACHER_API_KEY="${NANOHORIZON_TEACHER_API_KEY:-${NANOHORIZON_VLLM_API_KEY:-dummy-local-key}}"
export NANOHORIZON_MODAL_TEACHER_LAUNCH_MODE="${NANOHORIZON_MODAL_TEACHER_LAUNCH_MODE:-deploy}"
export NANOHORIZON_CRAFTAX_LOCAL_SHIMS="${NANOHORIZON_CRAFTAX_LOCAL_SHIMS:-2}"

source "$ROOT/scripts/lib_craftax_tunnel.sh"

cleanup() {
  nanohorizon_cleanup_craftax_tunnel
}
trap cleanup EXIT

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${NANOHORIZON_CRAFTAX_CONTAINER_URL:-}" ]]; then
  if [[ "$USE_LOCAL_DIRECT" == "1" ]]; then
    export NANOHORIZON_CRAFTAX_CONTAINER_URL="direct://local"
  else
    nanohorizon_start_local_craftax_if_needed "$ROOT"
    export NANOHORIZON_CRAFTAX_CONTAINER_URL="$(nanohorizon_craftax_local_urls)"
  fi
fi

nanohorizon_start_modal_teacher_if_needed "$ROOT"
TEACHER_BASE_URL="${NANOHORIZON_TEACHER_INFERENCE_URL%/v1/chat/completions}"
nanohorizon_wait_for_openai_compat_endpoint "$TEACHER_BASE_URL" "$NANOHORIZON_TEACHER_API_KEY" 240 2 1

exec uv run python -m nanohorizon.baselines.cpt_data run \
  --config "$CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
