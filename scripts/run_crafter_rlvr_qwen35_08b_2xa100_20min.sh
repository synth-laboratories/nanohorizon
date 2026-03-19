#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${NANOHORIZON_RLVR_CONFIG:-$ROOT/configs/crafter_rlvr_qwen35_08b_2xa100_20min.yaml}"
OUTPUT_ROOT="${NANOHORIZON_RLVR_OUTPUT_ROOT:-$ROOT/artifacts/rlvr_baseline}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${NANOHORIZON_AUTO_INSTALL:-0}" == "1" ]]; then
  python3 -m pip install -q "httpx>=0.28.1" "pyyaml>=6.0.2" "accelerate>=1.10.0" "peft>=0.17.0" "transformers>=4.57.0"
fi

echo "NanoHorizon RLVR bootstrap"
echo "  repo: $ROOT"
echo "  config: $CONFIG_PATH"
echo "  budget target: 20 minutes on 2x A100 40GB"
echo "  baseline: reward-weighted LoRA"

python3 -m nanohorizon.baselines.rlvr --config "$CONFIG_PATH" --output-dir "$OUTPUT_ROOT" "$@"
