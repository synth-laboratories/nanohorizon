#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${NANOHORIZON_OFFLINE_CONFIG:-$ROOT/configs/crafter_offline_qwen35_08b_1xa100_20min.yaml}"
OUTPUT_ROOT="${NANOHORIZON_OFFLINE_OUTPUT_ROOT:-$ROOT/artifacts/offline_sft_baseline}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
TEACHER_MODEL="${NANOHORIZON_TEACHER_MODEL:-Qwen/Qwen3.5-27B}"
TEACHER_PORT="${NANOHORIZON_TEACHER_PORT:-8001}"
TEACHER_BASE_URL_DEFAULT="${NANOHORIZON_TEACHER_BASE_URL:-http://127.0.0.1:${TEACHER_PORT}/v1}"
START_LOCAL_TEACHER="${NANOHORIZON_START_LOCAL_TEACHER:-0}"
TEACHER_STARTUP_ATTEMPTS="${NANOHORIZON_TEACHER_STARTUP_ATTEMPTS:-180}"
TEACHER_STARTUP_SLEEP_SECONDS="${NANOHORIZON_TEACHER_STARTUP_SLEEP_SECONDS:-2}"
TEACHER_LOG="$OUTPUT_ROOT/vllm_teacher.log"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

if [[ "${NANOHORIZON_AUTO_INSTALL:-0}" == "1" ]]; then
  log "installing Python runtime dependencies"
  python3 -m pip install "httpx>=0.28.1" "pyyaml>=6.0.2" "accelerate>=1.10.0" "datasets>=4.1.0" "peft>=0.17.0" "transformers>=4.57.0" "trl>=0.21.0" "vllm>=0.10.0"
  log "finished installing Python runtime dependencies"
fi

mkdir -p "$OUTPUT_ROOT"

cleanup() {
  if [[ -n "${TEACHER_PID:-}" ]]; then
    kill "$TEACHER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

log "NanoHorizon offline bootstrap"
log "repo: $ROOT"
log "config: $CONFIG_PATH"
log "budget target: 20 minutes on 1x A100 40GB"
log "baseline: vLLM teacher -> heuristic filter -> TRL SFT on 0.8B -> eval"

if [[ "$START_LOCAL_TEACHER" == "1" ]]; then
  export NANOHORIZON_TEACHER_BASE_URL="$TEACHER_BASE_URL_DEFAULT"
  export NANOHORIZON_TEACHER_API_KEY="${NANOHORIZON_TEACHER_API_KEY:-dummy-local-key}"
  log "starting teacher: $TEACHER_MODEL"
  log "teacher base url: $NANOHORIZON_TEACHER_BASE_URL"
  log "teacher log: $TEACHER_LOG"
  log "teacher startup attempts: $TEACHER_STARTUP_ATTEMPTS"
  log "teacher startup sleep seconds: $TEACHER_STARTUP_SLEEP_SECONDS"
  vllm serve "$TEACHER_MODEL" \
    --host 127.0.0.1 \
    --port "$TEACHER_PORT" \
    --api-key "$NANOHORIZON_TEACHER_API_KEY" \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.92 >"$TEACHER_LOG" 2>&1 &
  TEACHER_PID=$!
  for attempt in $(seq 1 "$TEACHER_STARTUP_ATTEMPTS"); do
    if curl -sf -H "Authorization: Bearer $NANOHORIZON_TEACHER_API_KEY" "http://127.0.0.1:${TEACHER_PORT}/v1/models" >/dev/null 2>&1; then
      log "teacher ready after $(( attempt * TEACHER_STARTUP_SLEEP_SECONDS )) seconds"
      break
    fi
    if (( attempt % 10 == 0 )); then
      log "waiting for teacher startup... attempt=$attempt"
      tail -n 20 "$TEACHER_LOG" 2>/dev/null || true
    fi
    sleep "$TEACHER_STARTUP_SLEEP_SECONDS"
  done
  if ! curl -sf -H "Authorization: Bearer $NANOHORIZON_TEACHER_API_KEY" "http://127.0.0.1:${TEACHER_PORT}/v1/models" >/dev/null; then
    log "teacher failed to become ready"
    tail -n 80 "$TEACHER_LOG" 2>/dev/null || true
    exit 1
  fi
fi

log "starting offline SFT baseline"
python3 -m nanohorizon.baselines.offline_sft --config "$CONFIG_PATH" --output-dir "$OUTPUT_ROOT" "$@"
log "offline SFT baseline finished"
