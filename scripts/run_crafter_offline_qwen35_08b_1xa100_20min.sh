#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${NANOHORIZON_OFFLINE_CONFIG:-$ROOT/configs/crafter_offline_qwen35_08b_1xa100_20min.yaml}"
OUTPUT_ROOT="${NANOHORIZON_OFFLINE_OUTPUT_ROOT:-$ROOT/artifacts/offline_sft_baseline}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
TEACHER_MODEL="${NANOHORIZON_TEACHER_MODEL:-Qwen/Qwen3.5-27B}"
TEACHER_BASE_URL_DEFAULT="${NANOHORIZON_TEACHER_BASE_URL:-http://127.0.0.1:8000/v1}"
START_LOCAL_TEACHER="${NANOHORIZON_START_LOCAL_TEACHER:-0}"
TEACHER_LOG="$OUTPUT_ROOT/vllm_teacher.log"

if [[ "${NANOHORIZON_AUTO_INSTALL:-0}" == "1" ]]; then
  python3 -m pip install -q "httpx>=0.28.1" "pyyaml>=6.0.2" "accelerate>=1.10.0" "datasets>=4.1.0" "peft>=0.17.0" "transformers>=4.57.0" "trl>=0.21.0" "vllm>=0.10.0"
fi

mkdir -p "$OUTPUT_ROOT"

cleanup() {
  if [[ -n "${TEACHER_PID:-}" ]]; then
    kill "$TEACHER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "NanoHorizon offline bootstrap"
echo "  repo: $ROOT"
echo "  config: $CONFIG_PATH"
echo "  budget target: 20 minutes on 1x A100 40GB"
echo "  baseline: vLLM 27B teacher -> heuristic filter -> TRL SFT on 0.8B -> eval"

if [[ "$START_LOCAL_TEACHER" == "1" ]]; then
  export NANOHORIZON_TEACHER_BASE_URL="$TEACHER_BASE_URL_DEFAULT"
  export NANOHORIZON_TEACHER_API_KEY="${NANOHORIZON_TEACHER_API_KEY:-dummy-local-key}"
  echo "  starting teacher: $TEACHER_MODEL"
  echo "  teacher base url: $NANOHORIZON_TEACHER_BASE_URL"
  echo "  teacher log: $TEACHER_LOG"
  vllm serve "$TEACHER_MODEL" \
    --host 127.0.0.1 \
    --port 8000 \
    --api-key "$NANOHORIZON_TEACHER_API_KEY" \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.92 >"$TEACHER_LOG" 2>&1 &
  TEACHER_PID=$!
  for attempt in $(seq 1 120); do
    if curl -sf -H "Authorization: Bearer $NANOHORIZON_TEACHER_API_KEY" http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
      echo "  teacher ready after $(( attempt * 2 )) seconds"
      break
    fi
    if (( attempt % 10 == 0 )); then
      echo "  waiting for teacher startup... attempt=$attempt"
      tail -n 20 "$TEACHER_LOG" 2>/dev/null || true
    fi
    sleep 2
  done
  if ! curl -sf -H "Authorization: Bearer $NANOHORIZON_TEACHER_API_KEY" http://127.0.0.1:8000/v1/models >/dev/null; then
    echo "  teacher failed to become ready"
    tail -n 80 "$TEACHER_LOG" 2>/dev/null || true
    exit 1
  fi
fi

python3 -m nanohorizon.baselines.offline_sft --config "$CONFIG_PATH" --output-dir "$OUTPUT_ROOT" "$@"
