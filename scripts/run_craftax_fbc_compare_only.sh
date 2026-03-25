#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_OUTPUT_DIR="${1:-${NANOHORIZON_REMOTE_OUTPUT_DIR:-}}"
if [[ -z "$REMOTE_OUTPUT_DIR" ]]; then
  echo "usage: $0 <remote_output_dir>" >&2
  exit 1
fi

ARTIFACT_DIR="${NANOHORIZON_COMPARE_ARTIFACT_DIR:-$ROOT/artifacts/fbc_compare_only_$(date -u +%Y%m%dT%H%M%SZ)}"
BASE_EVAL_DIR="$ARTIFACT_DIR/base_eval"
FINETUNED_EVAL_DIR="$ARTIFACT_DIR/finetuned_eval"
COMPARISON_PATH="$ARTIFACT_DIR/comparison_summary.json"
BASE_ENDPOINT_STDOUT="$ARTIFACT_DIR/base_endpoint.stdout.log"
BASE_ENDPOINT_STDERR="$ARTIFACT_DIR/base_endpoint.stderr.log"
FINETUNED_ENDPOINT_STDOUT="$ARTIFACT_DIR/finetuned_endpoint.stdout.log"
FINETUNED_ENDPOINT_STDERR="$ARTIFACT_DIR/finetuned_endpoint.stderr.log"

STUDENT_MODEL="${NANOHORIZON_STUDENT_MODEL:-Qwen/Qwen3.5-4B}"
THINKING_BUDGET="${NANOHORIZON_THINKING_BUDGET_TOKENS:-2000}"
MAX_NEW_TOKENS="${NANOHORIZON_MAX_NEW_TOKENS:-2200}"
MAX_MODEL_LEN="${NANOHORIZON_MAX_MODEL_LEN:-8192}"
NUM_EVAL_ROLLOUTS="${NANOHORIZON_NUM_EVAL_ROLLOUTS:-8}"
EVAL_SEED_START="${NANOHORIZON_EVAL_SEED_START:-10000}"
EVAL_MAX_STEPS="${NANOHORIZON_EVAL_MAX_STEPS:-8}"
EVAL_MAX_CONCURRENCY="${NANOHORIZON_EVAL_MAX_CONCURRENCY:-4}"
EVAL_REQUEST_TIMEOUT_SECONDS="${NANOHORIZON_EVAL_REQUEST_TIMEOUT_SECONDS:-600}"
VLLM_API_KEY="${NANOHORIZON_VLLM_API_KEY:-dummy-local-key}"
UV_BIN="${UV_BIN:-/opt/homebrew/bin/uv}"
ADAPTER_DIR="${REMOTE_OUTPUT_DIR%/}/adapter"

mkdir -p "$ARTIFACT_DIR" "$BASE_EVAL_DIR" "$FINETUNED_EVAL_DIR"

source "$ROOT/scripts/lib_craftax_tunnel.sh"

cleanup() {
  if [[ -n "${BASE_ENDPOINT_PID:-}" ]]; then
    kill "$BASE_ENDPOINT_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${FINETUNED_ENDPOINT_PID:-}" ]]; then
    kill "$FINETUNED_ENDPOINT_PID" >/dev/null 2>&1 || true
  fi
  nanohorizon_cleanup_craftax_tunnel
}
trap cleanup EXIT

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
    "$UV_BIN" run --group modal modal run src/nanohorizon/shared/modal_teacher.py --keepalive-s 7200 >"$log_stdout" 2>"$log_stderr" &
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
        if ! nanohorizon_wait_for_openai_compat_endpoint "$url" "$VLLM_API_KEY" 180 2; then
          echo "modal endpoint failed readiness probe: $url" >&2
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
  local -a cmd=(
    "$UV_BIN" run python -m nanohorizon.shared.eval_model
    --base-model "$STUDENT_MODEL"
    --container-url "${NANOHORIZON_CRAFTAX_CONTAINER_URL:-$NANOHORIZON_CRAFTAX_CONTAINER_URL}"
    --output-dir "$output_dir"
    --seed-start "$EVAL_SEED_START"
    --num-rollouts "$NUM_EVAL_ROLLOUTS"
    --max-steps "$EVAL_MAX_STEPS"
    --max-concurrent-rollouts "$EVAL_MAX_CONCURRENCY"
    --max-length "$MAX_MODEL_LEN"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --thinking-budget-tokens "$THINKING_BUDGET"
    --request-timeout-seconds "$EVAL_REQUEST_TIMEOUT_SECONDS"
    --inference-url "$inference_url"
    --inference-api-key "$VLLM_API_KEY"
    --request-model "$request_model"
  )
  if [[ "$THINKING_BUDGET" -gt 0 ]]; then
    cmd+=(--enable-thinking)
  fi
  "${cmd[@]}"
}

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
nanohorizon_load_synth_auth
nanohorizon_open_craftax_tunnel_if_needed "$ROOT"

base_endpoint_raw="$(start_modal_endpoint \
  "$BASE_ENDPOINT_STDOUT" \
  "$BASE_ENDPOINT_STDERR" \
  nanohorizon-craftax-student-base \
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
  nanohorizon-craftax-student-finetuned \
  "$STUDENT_MODEL" \
  "$MAX_MODEL_LEN" \
  "policy-lora" \
  "$ADAPTER_DIR" \
  "16")"
FINETUNED_ENDPOINT_PID="$(printf '%s\n' "$finetuned_endpoint_raw" | sed -n '1p')"
FINETUNED_INFERENCE_URL="$(printf '%s\n' "$finetuned_endpoint_raw" | sed -n '2p')"
run_local_eval "$FINETUNED_EVAL_DIR" "$FINETUNED_INFERENCE_URL" "policy-lora"

python3 - <<'PY' "$BASE_EVAL_DIR/eval_summary.json" "$FINETUNED_EVAL_DIR/eval_summary.json" "$COMPARISON_PATH" "$REMOTE_OUTPUT_DIR"
from pathlib import Path
import json
import sys

base_summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
finetuned_summary = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
comparison = {
    "remote_output_dir": sys.argv[4],
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
