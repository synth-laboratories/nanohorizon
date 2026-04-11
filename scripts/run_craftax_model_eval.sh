#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${NANOHORIZON_PROMPT_OPT_CONFIG:-configs/craftax_prompt_opt_qwen35_4b_full_auto_e2e.yaml}"
MANIFEST_PATH="${NANOHORIZON_CRAFTAX_CANDIDATE_MANIFEST:-experiments/nanohorizon_leaderboard_candidate/results/candidate_manifest.json}"
PROMPT_PATH="${NANOHORIZON_CRAFTAX_CANDIDATE_PROMPT:-experiments/nanohorizon_leaderboard_candidate/results/candidate_prompt.txt}"
OUTPUT_DIR="${NANOHORIZON_PROMPT_OPT_OUTPUT_DIR:-experiments/nanohorizon_leaderboard_candidate/results}"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"
exec uv run --with-editable . --python 3.12 python -m nanohorizon.craftax_core.runner \
  --config "$CONFIG_PATH" \
  --write "$MANIFEST_PATH" \
  --prompt-out "$PROMPT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
