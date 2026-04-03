#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/lib_craftax_tunnel.sh"

CONFIG_PATH="${NANOHORIZON_PROMPT_OPT_CONFIG:-configs/craftax_prompt_opt_gemini25_flash_lite_local_eval20.yaml}"
OUTPUT_DIR="${NANOHORIZON_PROMPT_OPT_OUTPUT_DIR:-}"
REQUEST_MODEL="${NANOHORIZON_PROMPT_OPT_REQUEST_MODEL:-gemini-2.5-flash-lite}"
GEMINI_OPENAI_BASE_URL="${NANOHORIZON_PROMPT_OPT_GEMINI_OPENAI_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai}"
EXPERIMENT_API_KEY="${GEMINI_API_KEY:-${OPENAI_API_KEY:-}}"
EXPERIMENT_BASE_URL="${NANOHORIZON_PROMPT_OPT_INFERENCE_BASE_URL:-${OPENAI_BASE_URL:-}}"

if [[ -z "$EXPERIMENT_BASE_URL" ]]; then
  if [[ -n "${GEMINI_API_KEY:-}" ]]; then
    EXPERIMENT_BASE_URL="$GEMINI_OPENAI_BASE_URL"
  else
    EXPERIMENT_BASE_URL="$GEMINI_OPENAI_BASE_URL"
  fi
fi

if [[ -z "$EXPERIMENT_API_KEY" ]]; then
  echo "missing GEMINI_API_KEY or OPENAI_API_KEY for local Gemini prompt-opt run" >&2
  exit 1
fi

cd "$ROOT"
nanohorizon_start_local_craftax_if_needed "$ROOT"

CONTAINER_URL="${NANOHORIZON_PROMPT_OPT_CONTAINER_URL:-http://127.0.0.1:$(nanohorizon_craftax_local_port)}"
INFERENCE_API_KEY="${NANOHORIZON_PROMPT_OPT_INFERENCE_API_KEY:-$EXPERIMENT_API_KEY}"

exec "${UV_BIN:-uv}" run --group modal python -m nanohorizon.baselines.prompt_opt \
  --config "$CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --container-url "$CONTAINER_URL" \
  --inference-url "$EXPERIMENT_BASE_URL" \
  --inference-api-key "$INFERENCE_API_KEY" \
  --request-model "$REQUEST_MODEL" \
  "$@"
