#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${NANOHORIZON_PROMPT_OPT_CONFIG:-configs/craftax_prompt_opt_qwen35_4b_gpt54_budget.yaml}"
OUTPUT_DIR="${NANOHORIZON_PROMPT_OPT_OUTPUT_DIR:-}"

cd "$ROOT"
exec uv run --group modal modal run src/nanohorizon/baselines/prompt_opt.py \
  --config "$CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
