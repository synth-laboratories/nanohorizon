#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${NANOHORIZON_PROMPT_OPT_CONFIG:-$ROOT/configs/crafter_prompt_opt_qwen35_08b_gpt54_budget.yaml}"
OUTPUT_ROOT="${NANOHORIZON_PROMPT_OPT_OUTPUT_ROOT:-$ROOT/artifacts/prompt_opt_baseline}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${RUNPOD_POD_ID:-}" || "${NANOHORIZON_AUTO_INSTALL:-0}" == "1" ]]; then
  python3 -m pip install -q "httpx>=0.28.1" "pyyaml>=6.0.2"
fi

mkdir -p "$OUTPUT_ROOT"

echo "NanoHorizon prompt optimization bootstrap"
echo "  repo: $ROOT"
echo "  config: $CONFIG_PATH"
echo "  optimizer budget: $1 total spend on gpt-5.4, gpt-5.4-mini, and gpt-5.4-nano"
echo "  evaluation policy: Qwen/Qwen3.5-0.8B"
echo "  baseline: GEPA-style prompt search"

python3 -m nanohorizon.baselines.prompt_opt --config "$CONFIG_PATH" --output-dir "$OUTPUT_ROOT" "$@"
