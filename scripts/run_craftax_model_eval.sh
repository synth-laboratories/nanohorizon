#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL="${NANOHORIZON_EVAL_BASE_MODEL:-Qwen/Qwen3.5-4B}"
ADAPTER_DIR="${NANOHORIZON_EVAL_ADAPTER_DIR:-}"
OUTPUT_DIR="${NANOHORIZON_EVAL_OUTPUT_DIR:-}"
UV_BIN="${UV_BIN:-$(command -v uv)}"
SEED_START="${NANOHORIZON_EVAL_SEED_START:-10000}"
NUM_ROLLOUTS="${NANOHORIZON_EVAL_NUM_ROLLOUTS:-8}"
MAX_STEPS="${NANOHORIZON_EVAL_MAX_STEPS:-48}"
MAX_CONCURRENT_ROLLOUTS="${NANOHORIZON_EVAL_MAX_CONCURRENT_ROLLOUTS:-8}"
MAX_LENGTH="${NANOHORIZON_EVAL_MAX_LENGTH:-4096}"
MAX_NEW_TOKENS="${NANOHORIZON_EVAL_MAX_NEW_TOKENS:-1024}"
THINKING_BUDGET_TOKENS="${NANOHORIZON_EVAL_THINKING_BUDGET_TOKENS:-512}"
ENABLE_THINKING="${NANOHORIZON_EVAL_ENABLE_THINKING:-0}"
ENFORCE_EAGER="${NANOHORIZON_EVAL_ENFORCE_EAGER:-0}"

cd "$ROOT"
source "$ROOT/scripts/lib_craftax_tunnel.sh"
trap nanohorizon_cleanup_craftax_tunnel EXIT
nanohorizon_open_craftax_tunnel_if_needed "$ROOT"

cmd=(
  "$UV_BIN" run --group modal modal run src/nanohorizon/shared/modal_eval.py
  --base-model "$BASE_MODEL"
  --adapter-dir "$ADAPTER_DIR"
  --container-url "${NANOHORIZON_CRAFTAX_CONTAINER_URL:-${NANOHORIZON_CRAFTAX_CONTAINER_URL:-}}"
  --container-worker-token "${NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN:-${NANOHORIZON_CRAFTAX_CONTAINER_WORKER_TOKEN:-}}"
  --output-dir "$OUTPUT_DIR"
  --seed-start "$SEED_START"
  --num-rollouts "$NUM_ROLLOUTS"
  --max-steps "$MAX_STEPS"
  --max-concurrent-rollouts "$MAX_CONCURRENT_ROLLOUTS"
  --max-length "$MAX_LENGTH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --thinking-budget-tokens "$THINKING_BUDGET_TOKENS"
)
if [[ "$ENABLE_THINKING" == "1" ]]; then
  cmd+=(--enable-thinking)
fi
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  cmd+=(--enforce-eager)
fi
cmd+=("$@")
"${cmd[@]}"
