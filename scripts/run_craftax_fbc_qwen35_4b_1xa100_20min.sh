#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NANOHORIZON_OFFLINE_CONFIG="${NANOHORIZON_FBC_CONFIG:-configs/craftax_fbc_qwen35_4b_1xa100_20min.yaml}"
export NANOHORIZON_OFFLINE_OUTPUT_DIR="${NANOHORIZON_FBC_OUTPUT_DIR:-}"
exec "$ROOT/scripts/run_craftax_offline_qwen35_4b_1xa100_20min.sh" "$@"
