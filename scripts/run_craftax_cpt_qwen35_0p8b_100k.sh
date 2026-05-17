#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${NANOHORIZON_CPT_CONFIG:-configs/craftax_cpt_qwen35_0p8b_100k.yaml}"
OUTPUT_DIR="${NANOHORIZON_CPT_OUTPUT_DIR:-$ROOT/artifacts/cpt_qwen35_0p8b_100k_$(date -u +%Y%m%dT%H%M%SZ)}"
CPT_BACKEND="${NANOHORIZON_CPT_BACKEND:-modal}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$CPT_BACKEND" == "local" ]]; then
  exec uv run python -m nanohorizon.baselines.cpt hf-run \
    --config "$CONFIG_PATH" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
else
  exec uv run --group modal modal run src/nanohorizon/baselines/cpt.py \
    --config "$CONFIG_PATH" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
fi
