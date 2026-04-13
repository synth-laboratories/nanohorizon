#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${NANOHORIZON_OFFLINE_CONFIG:-$ROOT/configs/craftax_offline_reference.yaml}"
OUTPUT_ROOT="${NANOHORIZON_OFFLINE_OUTPUT_ROOT:-$ROOT/artifacts/offline_reference}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
CODE_VERSION="${NANOHORIZON_CODE_VERSION:-unknown}"
APP_NAME="${NANOHORIZON_MODAL_OFFLINE_APP_NAME:-nanohorizon-craftax-offline}"
TEACHER_MODEL="${NANOHORIZON_TEACHER_MODEL:-Qwen/Qwen3.5-9B}"
TEACHER_PORT="${NANOHORIZON_TEACHER_PORT:-8001}"
TEACHER_MAX_MODEL_LEN="${NANOHORIZON_TEACHER_MAX_MODEL_LEN:-8192}"
TEACHER_MAX_NUM_SEQS="${NANOHORIZON_TEACHER_MAX_NUM_SEQS:-64}"
TEACHER_GPU_MEMORY_UTILIZATION="${NANOHORIZON_TEACHER_GPU_MEMORY_UTILIZATION:-0.92}"
TEACHER_REASONING_PARSER="${NANOHORIZON_TEACHER_REASONING_PARSER:-qwen3}"
TEACHER_TOOL_CALL_PARSER="${NANOHORIZON_TEACHER_TOOL_CALL_PARSER:-qwen3_coder}"
TEACHER_ENFORCE_EAGER="${NANOHORIZON_TEACHER_ENFORCE_EAGER:-1}"
TEACHER_BASE_URL_DEFAULT="${NANOHORIZON_TEACHER_BASE_URL:-http://127.0.0.1:${TEACHER_PORT}/v1}"
START_LOCAL_TEACHER="${NANOHORIZON_START_LOCAL_TEACHER:-0}"
TEACHER_STARTUP_ATTEMPTS="${NANOHORIZON_TEACHER_STARTUP_ATTEMPTS:-240}"
TEACHER_STARTUP_SLEEP_SECONDS="${NANOHORIZON_TEACHER_STARTUP_SLEEP_SECONDS:-2}"
TEACHER_LOG="$OUTPUT_ROOT/vllm_teacher.log"
VENV_ROOT="${NANOHORIZON_VENV_ROOT:-/opt/nanohorizon-offline-venvs}"
TEACHER_VENV="$VENV_ROOT/teacher"
PYTHON_BIN="${PYTHON_BIN:-}"
VLLM_BIN="${VLLM_BIN:-}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

if [[ -z "$VLLM_BIN" ]]; then
  if [[ -x "$TEACHER_VENV/bin/vllm" ]]; then
    VLLM_BIN="$TEACHER_VENV/bin/vllm"
  else
    VLLM_BIN="vllm"
  fi
fi

if [[ "${NANOHORIZON_AUTO_INSTALL:-0}" == "1" ]]; then
  mkdir -p "$VENV_ROOT"

  if [[ ! -x "$TEACHER_VENV/bin/python" ]]; then
    log "creating teacher virtualenv"
    python3 -m venv --system-site-packages "$TEACHER_VENV"
    "$TEACHER_VENV/bin/python" -m pip install --upgrade pip
    "$TEACHER_VENV/bin/python" -m pip install \
      "httpx>=0.28.1" \
      "pyyaml>=6.0.2" \
      "vllm>=0.10.0"
  else
    log "reusing teacher virtualenv"
  fi

  VLLM_BIN="$TEACHER_VENV/bin/vllm"
  log "finished preparing offline runtime dependencies"
elif [[ "$START_LOCAL_TEACHER" == "1" && ! -x "$TEACHER_VENV/bin/vllm" ]]; then
  log "prebuilt teacher runtime not found under $VENV_ROOT"
  log "set NANOHORIZON_AUTO_INSTALL=1 to build them at runtime"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

cleanup() {
  if [[ -n "${TEACHER_PGID:-}" ]]; then
    kill -- "-$TEACHER_PGID" >/dev/null 2>&1 || true
  elif [[ -n "${TEACHER_PID:-}" ]]; then
    kill "$TEACHER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

log "NanoHorizon offline bootstrap"
log "modal app: $APP_NAME"
log "code version: $CODE_VERSION"
log "repo: $ROOT"
log "config: $CONFIG_PATH"
log "budget target: 20 minutes on 1x A100 40GB"
log "baseline: async parallel rollout collection -> reward filter -> TRL SFT on 4B -> base vs finetuned eval"

if [[ "$START_LOCAL_TEACHER" == "1" ]]; then
  export NANOHORIZON_TEACHER_BASE_URL="$TEACHER_BASE_URL_DEFAULT"
  export NANOHORIZON_TEACHER_API_KEY="${NANOHORIZON_TEACHER_API_KEY:-dummy-local-key}"
  log "starting teacher: $TEACHER_MODEL"
  log "teacher base url: $NANOHORIZON_TEACHER_BASE_URL"
  log "teacher log: $TEACHER_LOG"
  log "teacher max model len: $TEACHER_MAX_MODEL_LEN"
  log "teacher max num seqs: $TEACHER_MAX_NUM_SEQS"
  log "teacher gpu memory utilization: $TEACHER_GPU_MEMORY_UTILIZATION"
  log "teacher enforce eager: $TEACHER_ENFORCE_EAGER"
  log "teacher startup attempts: $TEACHER_STARTUP_ATTEMPTS"
  log "teacher startup sleep seconds: $TEACHER_STARTUP_SLEEP_SECONDS"
  teacher_cmd=(
    "$VLLM_BIN" serve "$TEACHER_MODEL"
    --host 127.0.0.1
    --port "$TEACHER_PORT"
    --api-key "$NANOHORIZON_TEACHER_API_KEY"
    --max-model-len "$TEACHER_MAX_MODEL_LEN"
    --max-num-seqs "$TEACHER_MAX_NUM_SEQS"
    --gpu-memory-utilization "$TEACHER_GPU_MEMORY_UTILIZATION"
    --uvicorn-log-level info
    --enable-prefix-caching
    --language-model-only
    --reasoning-parser "$TEACHER_REASONING_PARSER"
    --logits-processors nanohorizon.custom_vllm.thinking_budget:QwenThinkingBudgetLogitsProcessor
    --enable-auto-tool-choice
    --tool-call-parser "$TEACHER_TOOL_CALL_PARSER"
  )
  if [[ "$TEACHER_ENFORCE_EAGER" == "1" ]]; then
    teacher_cmd+=(--enforce-eager)
  fi
  setsid "${teacher_cmd[@]}" >"$TEACHER_LOG" 2>&1 &
  TEACHER_PID=$!
  TEACHER_PGID="$TEACHER_PID"
  export NANOHORIZON_LOCAL_TEACHER_PID="$TEACHER_PID"
  export NANOHORIZON_LOCAL_TEACHER_PGID="$TEACHER_PGID"
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

log "starting teacher rollout collection"
"$PYTHON_BIN" -m nanohorizon.baselines.offline_sft --config "$CONFIG_PATH" --output-dir "$OUTPUT_ROOT" "$@"
log "offline training finished"
