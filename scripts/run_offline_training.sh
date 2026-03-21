#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${NANOHORIZON_OFFLINE_TRAINING_ARTIFACT_DIR:-$ROOT/artifacts/offline_reference_$(date -u +%Y%m%dT%H%M%SZ)}"
CONFIG_PATH="${NANOHORIZON_OFFLINE_CONFIG:-configs/crafter_offline_reference.yaml}"
TRAINING_LOG="$ARTIFACT_DIR/offline_training.log"
MODAL_SFT_DEPLOY_LOG="$ARTIFACT_DIR/modal_sft_deploy.log"
MODAL_TRAIN_RESULT_PATH="$ARTIFACT_DIR/modal_train_result.json"
BASE_ENDPOINT_STDOUT="$ARTIFACT_DIR/base_endpoint.stdout.log"
BASE_ENDPOINT_STDERR="$ARTIFACT_DIR/base_endpoint.stderr.log"
FINETUNED_ENDPOINT_STDOUT="$ARTIFACT_DIR/finetuned_endpoint.stdout.log"
FINETUNED_ENDPOINT_STDERR="$ARTIFACT_DIR/finetuned_endpoint.stderr.log"
BASE_EVAL_DIR="$ARTIFACT_DIR/base_eval"
FINETUNED_EVAL_DIR="$ARTIFACT_DIR/finetuned_eval"
COMPARISON_PATH="$ARTIFACT_DIR/comparison_summary.json"

STUDENT_MODEL="${NANOHORIZON_STUDENT_MODEL:-Qwen/Qwen3.5-4B}"
TEACHER_MODEL="${NANOHORIZON_TEACHER_MODEL:-Qwen/Qwen3.5-9B}"
THINKING_BUDGET="${NANOHORIZON_THINKING_BUDGET_TOKENS:-2000}"
MAX_NEW_TOKENS="${NANOHORIZON_MAX_NEW_TOKENS:-2200}"
MAX_MODEL_LEN="${NANOHORIZON_MAX_MODEL_LEN:-8192}"
NUM_EVAL_ROLLOUTS="${NANOHORIZON_NUM_EVAL_ROLLOUTS:-20}"
EVAL_SEED_START="${NANOHORIZON_EVAL_SEED_START:-10000}"
EVAL_MAX_STEPS="${NANOHORIZON_EVAL_MAX_STEPS:-8}"
EVAL_MAX_CONCURRENCY="${NANOHORIZON_EVAL_MAX_CONCURRENCY:-10}"
EVAL_REQUEST_TIMEOUT_SECONDS="${NANOHORIZON_EVAL_REQUEST_TIMEOUT_SECONDS:-600}"
VLLM_API_KEY="${NANOHORIZON_VLLM_API_KEY:-dummy-local-key}"
UV_BIN="${UV_BIN:-/opt/homebrew/bin/uv}"
DEPLOY_BEFORE_RUN="${NANOHORIZON_MODAL_DEPLOY_BEFORE_RUN:-1}"
TRAIN_ON_MODAL="${NANOHORIZON_TRAIN_ON_MODAL:-1}"

mkdir -p "$ARTIFACT_DIR" "$BASE_EVAL_DIR" "$FINETUNED_EVAL_DIR"

source "$ROOT/scripts/lib_crafter_tunnel.sh"

cleanup() {
  if [[ -n "${BASE_ENDPOINT_PID:-}" ]]; then
    kill "$BASE_ENDPOINT_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${FINETUNED_ENDPOINT_PID:-}" ]]; then
    kill "$FINETUNED_ENDPOINT_PID" >/dev/null 2>&1 || true
  fi
  nanohorizon_cleanup_crafter_tunnel
}
trap cleanup EXIT

wait_for_modal_endpoint() {
  local base_url="$1"
  local api_key="$2"

  for _ in $(seq 1 180); do
    if curl --max-time 10 -sf -H "Authorization: Bearer $api_key" "${base_url%/}/v1/models" >/dev/null 2>&1; then
      python3 - <<'PY' "${base_url%/}/v1/chat/completions" "$api_key" >/dev/null 2>&1 || true
import json
import sys
import urllib.request

url = sys.argv[1]
api_key = sys.argv[2]
payload = {
    "model": "Qwen/Qwen3.5-4B",
    "messages": [{"role": "user", "content": "ping"}],
    "max_tokens": 8,
    "temperature": 0.0,
}
request = urllib.request.Request(
    url,
    method="POST",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
)
with urllib.request.urlopen(request, timeout=30):
    pass
PY
      return 0
    fi
    sleep 2
  done

  return 1
}

start_modal_endpoint() {
  local log_stdout="$1"
  local log_stderr="$2"
  local app_name="$3"
  local model_ref="$4"
  local max_model_len="$5"
  local lora_name="$6"
  local lora_path="$7"
  local max_lora_rank="$8"

  env \
    NANOHORIZON_MODAL_TEACHER_APP_NAME="$app_name" \
    NANOHORIZON_TEACHER_MODEL="$model_ref" \
    NANOHORIZON_TEACHER_API_KEY="$VLLM_API_KEY" \
    NANOHORIZON_TEACHER_MAX_MODEL_LEN="$max_model_len" \
    NANOHORIZON_TEACHER_LORA_NAME="$lora_name" \
    NANOHORIZON_TEACHER_LORA_PATH="$lora_path" \
    NANOHORIZON_TEACHER_MAX_LORA_RANK="$max_lora_rank" \
    "$UV_BIN" run --group modal modal run src/nanohorizon/modal_teacher.py --keepalive-s 7200 >"$log_stdout" 2>"$log_stderr" &
  local pid=$!
  for _ in $(seq 1 240); do
    if [[ -f "$log_stdout" ]]; then
      local url
      url="$(python3 - <<'PY' "$log_stdout"
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8") if path.exists() else ""
for raw in text.splitlines():
    if raw.startswith("vLLM endpoint: "):
        print(raw.split("vLLM endpoint: ", 1)[1].strip())
        break
PY
)"
      if [[ -n "$url" ]]; then
        if ! wait_for_modal_endpoint "${url%/}" "$VLLM_API_KEY"; then
          cat "$log_stderr" >&2 || true
          cat "$log_stdout" >&2 || true
          return 1
        fi
        printf '%s\n%s\n' "$pid" "${url%/}/v1/chat/completions"
        return 0
      fi
    fi
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      cat "$log_stderr" >&2 || true
      cat "$log_stdout" >&2 || true
      return 1
    fi
    sleep 2
  done

  cat "$log_stderr" >&2 || true
  cat "$log_stdout" >&2 || true
  return 1
}

run_local_eval() {
  local output_dir="$1"
  local inference_url="$2"
  local request_model="$3"
  "$UV_BIN" run python -m nanohorizon.eval_model \
    --base-model "$STUDENT_MODEL" \
    --container-url "$NANOHORIZON_CRAFTER_CONTAINER_URL" \
    --output-dir "$output_dir" \
    --seed-start "$EVAL_SEED_START" \
    --num-rollouts "$NUM_EVAL_ROLLOUTS" \
    --max-steps "$EVAL_MAX_STEPS" \
    --max-concurrent-rollouts "$EVAL_MAX_CONCURRENCY" \
    --max-length "$MAX_MODEL_LEN" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --thinking-budget-tokens "$THINKING_BUDGET" \
    --enable-thinking \
    --request-timeout-seconds "$EVAL_REQUEST_TIMEOUT_SECONDS" \
    --inference-url "$inference_url" \
    --inference-api-key "$VLLM_API_KEY" \
    --request-model "$request_model"
}

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export NANOHORIZON_OFFLINE_CONFIG="$CONFIG_PATH"
export NANOHORIZON_TEACHER_MODEL="$TEACHER_MODEL"
export NANOHORIZON_TEACHER_MAX_MODEL_LEN="$MAX_MODEL_LEN"
export NANOHORIZON_TEACHER_ENFORCE_EAGER="${NANOHORIZON_TEACHER_ENFORCE_EAGER:-1}"
export NANOHORIZON_MIN_TEACHER_REWARD="${NANOHORIZON_MIN_TEACHER_REWARD:-1.0}"
export NANOHORIZON_MAX_TEACHER_ROWS="${NANOHORIZON_MAX_TEACHER_ROWS:-32}"
export NANOHORIZON_FILTER_COLLECT_WOOD="${NANOHORIZON_FILTER_COLLECT_WOOD:-1}"
export NANOHORIZON_MODAL_GPU_OFFLINE="${NANOHORIZON_MODAL_GPU_OFFLINE:-A100-40GB}"
export NANOHORIZON_TRAIN_ON_MODAL="$TRAIN_ON_MODAL"
export NANOHORIZON_MODAL_SFT_APP_NAME="${NANOHORIZON_MODAL_SFT_APP_NAME:-nanohorizon-crafter-sft}"
export NANOHORIZON_MODAL_SFT_OUTPUT_DIR="${NANOHORIZON_MODAL_SFT_OUTPUT_DIR:-/vol/artifacts/offline_reference/$(basename "$ARTIFACT_DIR")}"

nanohorizon_start_local_crafter_if_needed "$ROOT"
export NANOHORIZON_CRAFTER_CONTAINER_URL="${NANOHORIZON_CRAFTER_CONTAINER_URL:-http://127.0.0.1:8903}"
unset NANOHORIZON_CRAFTER_CONTAINER_WORKER_TOKEN

nanohorizon_start_modal_teacher_if_needed "$ROOT"
if [[ "$TRAIN_ON_MODAL" == "1" && "$DEPLOY_BEFORE_RUN" == "1" ]]; then
  "$UV_BIN" run --group modal modal deploy src/nanohorizon/modal_sft.py >"$MODAL_SFT_DEPLOY_LOG" 2>&1
fi

"$UV_BIN" run --group modal python -m nanohorizon.offline_training \
  --config "$CONFIG_PATH" \
  --output-dir "$ARTIFACT_DIR" | tee "$TRAINING_LOG"

ADAPTER_DIR="$(python3 - <<'PY' "$MODAL_TRAIN_RESULT_PATH" "$ARTIFACT_DIR"
from pathlib import Path
import json
import sys

modal_result_path = Path(sys.argv[1])
artifact_dir = Path(sys.argv[2]).resolve()
if modal_result_path.exists():
    payload = json.loads(modal_result_path.read_text(encoding="utf-8"))
    value = str(payload.get("adapter_dir") or "").strip()
    if value:
        print(value)
        raise SystemExit(0)
local_adapter = artifact_dir / "adapter"
if local_adapter.exists():
    print(str(local_adapter))
    raise SystemExit(0)
raise SystemExit(1)
PY
)"

base_endpoint_raw="$(start_modal_endpoint \
  "$BASE_ENDPOINT_STDOUT" \
  "$BASE_ENDPOINT_STDERR" \
  nanohorizon-crafter-student-base \
  "$STUDENT_MODEL" \
  "$MAX_MODEL_LEN" \
  "" \
  "" \
  "16")"
BASE_ENDPOINT_PID="$(printf '%s\n' "$base_endpoint_raw" | sed -n '1p')"
BASE_INFERENCE_URL="$(printf '%s\n' "$base_endpoint_raw" | sed -n '2p')"
run_local_eval "$BASE_EVAL_DIR" "$BASE_INFERENCE_URL" "$STUDENT_MODEL"
kill "$BASE_ENDPOINT_PID" >/dev/null 2>&1 || true
unset BASE_ENDPOINT_PID

finetuned_endpoint_raw="$(start_modal_endpoint \
  "$FINETUNED_ENDPOINT_STDOUT" \
  "$FINETUNED_ENDPOINT_STDERR" \
  nanohorizon-crafter-student-finetuned \
  "$STUDENT_MODEL" \
  "$MAX_MODEL_LEN" \
  "policy-lora" \
  "$ADAPTER_DIR" \
  "16")"
FINETUNED_ENDPOINT_PID="$(printf '%s\n' "$finetuned_endpoint_raw" | sed -n '1p')"
FINETUNED_INFERENCE_URL="$(printf '%s\n' "$finetuned_endpoint_raw" | sed -n '2p')"
run_local_eval "$FINETUNED_EVAL_DIR" "$FINETUNED_INFERENCE_URL" "policy-lora"

python3 - <<'PY' "$BASE_EVAL_DIR/eval_summary.json" "$FINETUNED_EVAL_DIR/eval_summary.json" "$COMPARISON_PATH" "$ADAPTER_DIR"
from pathlib import Path
import json
import sys

base_summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
finetuned_summary = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
comparison = {
    "adapter_dir": sys.argv[4],
    "base_mean_outcome_reward": float(base_summary.get("mean_outcome_reward", 0.0) or 0.0),
    "finetuned_mean_outcome_reward": float(finetuned_summary.get("mean_outcome_reward", 0.0) or 0.0),
    "reward_delta": float(finetuned_summary.get("mean_outcome_reward", 0.0) or 0.0)
    - float(base_summary.get("mean_outcome_reward", 0.0) or 0.0),
    "base_num_eval_rollouts": int(base_summary.get("num_eval_rollouts", 0) or 0),
    "finetuned_num_eval_rollouts": int(finetuned_summary.get("num_eval_rollouts", 0) or 0),
}
Path(sys.argv[3]).write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(comparison, indent=2, sort_keys=True))
PY
