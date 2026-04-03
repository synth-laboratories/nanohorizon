#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NANOHORIZON_RLVR_CONFIG="${NANOHORIZON_RLVR_CONFIG:-configs/craftax_rlvr_qwen35_0p8b_2xa100_20min.yaml}"

exec "$ROOT/scripts/run_craftax_rlvr_qwen35_4b_2xa100_20min.sh" "$@"
