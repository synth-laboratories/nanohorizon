#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${NANOHORIZON_PIVOT_VERIFIER_CONFIG:-configs/pivot_verifier_qwen35_4b_spct.yaml}"
OUTPUT_DIR="${NANOHORIZON_PIVOT_VERIFIER_OUTPUT_DIR:-}"
APP_NAME="${NANOHORIZON_MODAL_PIVOT_VERIFIER_APP_NAME:-nanohorizon-craftax-pivot-verifier}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export NANOHORIZON_MODAL_PIVOT_VERIFIER_APP_NAME="$APP_NAME"
export NANOHORIZON_MODAL_GPU_OFFLINE="${NANOHORIZON_MODAL_GPU_OFFLINE:-L4}"

exec uv run --group modal modal run submissions/synth/pivot_verifier.py \
  --config "$CONFIG_PATH" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
