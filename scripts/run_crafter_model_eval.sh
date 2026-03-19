#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

BASE_MODEL="${NANOHORIZON_EVAL_BASE_MODEL:-Qwen/Qwen3.5-0.8B}"
ADAPTER_DIR="${NANOHORIZON_EVAL_ADAPTER_DIR:-}"
EVAL_JSONL="${NANOHORIZON_EVAL_JSONL:-$ROOT/data/crafter/crafter_eval_prompts.jsonl}"
OUTPUT_DIR="${NANOHORIZON_EVAL_OUTPUT_DIR:-$ROOT/artifacts/model_eval}"
MAX_LENGTH="${NANOHORIZON_EVAL_MAX_LENGTH:-512}"
MAX_NEW_TOKENS="${NANOHORIZON_EVAL_MAX_NEW_TOKENS:-32}"

ARGS=(
  --base-model "$BASE_MODEL"
  --eval-jsonl "$EVAL_JSONL"
  --output-dir "$OUTPUT_DIR"
  --max-length "$MAX_LENGTH"
  --max-new-tokens "$MAX_NEW_TOKENS"
)

if [[ -n "$ADAPTER_DIR" ]]; then
  ARGS+=(--adapter-dir "$ADAPTER_DIR")
fi

python3 -m nanohorizon.eval_model "${ARGS[@]}" "$@"
