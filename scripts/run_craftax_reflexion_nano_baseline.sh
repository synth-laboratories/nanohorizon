#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${NANOHORIZON_REFLEXION_CONFIG:-configs/craftax_reflexion_nano.yaml}"
OUTPUT_DIR="${NANOHORIZON_REFLEXION_OUTPUT_DIR:-}"
REQUEST_MODEL="${NANOHORIZON_REFLEXION_REQUEST_MODEL:-gpt-4.1-nano}"
INFERENCE_API_KEY="${NANOHORIZON_REFLEXION_INFERENCE_API_KEY:-${OPENAI_API_KEY:-}}"
INFERENCE_URL="${NANOHORIZON_REFLEXION_INFERENCE_URL:-}"
INFERENCE_BASE_URL="${NANOHORIZON_REFLEXION_INFERENCE_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com}}"

if [[ -z "${INFERENCE_URL}" ]]; then
  BASE="${INFERENCE_BASE_URL%/}"
  if [[ "${BASE}" == */v1 ]]; then
    INFERENCE_URL="${BASE}/chat/completions"
  else
    INFERENCE_URL="${BASE}/v1/chat/completions"
  fi
fi

if [[ -z "${INFERENCE_API_KEY}" ]]; then
  echo "NANOHORIZON_REFLEXION_INFERENCE_API_KEY or OPENAI_API_KEY is required." >&2
  exit 1
fi

cd "$ROOT"
ARGS=(
  --config "$CONFIG_PATH"
  --container-url "direct://local"
  --inference-url "$INFERENCE_URL"
  --request-model "$REQUEST_MODEL"
)
if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi

export OPENAI_API_KEY="$INFERENCE_API_KEY"
exec uv run --group modal --group classic python -m nanohorizon.baselines.prompt_opt \
  "${ARGS[@]}" \
  "$@"
