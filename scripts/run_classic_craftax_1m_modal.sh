#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${NANOHORIZON_CLASSIC_CONFIG:-configs/classic_craftax_1m_random_init.yaml}"
OUTPUT_DIR="${NANOHORIZON_CLASSIC_OUTPUT_DIR:-}"

cd "$ROOT"
exec uv run --group modal modal run src/nanohorizon/baselines/classic_modal.py \
  --config "$CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
