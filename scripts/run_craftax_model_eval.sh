#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/.out/craftax_eval}"

mkdir -p "${OUT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m nanohorizon.craftax_core.runner \
  --smoke \
  --json \
  --output "${OUT_DIR}/smoke_payload.json"
