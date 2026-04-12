#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${NANOHORIZON_RLVR_CONFIG:-configs/craftax_rlvr_qwen35_4b_2xa100_20min.yaml}"
OUTPUT_DIR="${NANOHORIZON_RLVR_OUTPUT_DIR:-}"

cd "$ROOT"
exec uv run --group modal --group classic --group training modal run src/nanohorizon/baselines/rlvr.py \
  --config "$CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
